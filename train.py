import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer 
from tokenizers.pre_tokenizers import Whitespace
from data_model.chat_data_model import ChatDataset, causal_mask
from torch.utils.tensorboard import SummaryWriter
from config import get_config, get_weights_file_path
from tqdm import tqdm
from model import build_transformer
from pathlib import Path
import warnings

def greedy_decode(model, source, source_mask, tokenizer, max_len, device):
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

    # Precompute the encoder input and reuse it for every token from the decoder
    encoder_output = model.encode(source, source_mask)
    # Initally the decoder input is the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1, :])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    console_width = 80
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device) # (batch_dim, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_dim, 1, 1, seq_len)

            assert encoder_input.size(0) == 1, "batch size should be 1"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())

            print_msg("-" * console_width)
            print_msg(f'model out text {model_out.detach().cpu().numpy()}')
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            if count == num_examples:
                break

def get_all_sentences(ds):
    for item in ds:
        yield item['source']

def get_or_build_tokenizer(config, ds):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # load_data set from the local file
    with open("datasets/chatbot_dataset.json", "r") as file:
        ds_raw = json.load(file)
    print("len of data examples ", len(ds_raw))
    ds_raw = ds_raw[:int(0.15 * len(ds_raw))]
    print("len of data examples ", len(ds_raw))
    
    # build tokenizer
    tokenizer = get_or_build_tokenizer(config, ds_raw)

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    test_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, test_ds_size])

    train_ds = ChatDataset(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = ChatDataset(val_ds_raw, tokenizer, config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer


def get_model(config, vocab_size):
    model = build_transformer(vocab_size, vocab_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)

    Path(config['model_foldername']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    # tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # optimizer
    optimizer = torch.optim.Adam(model.parameters() , lr=config['lr'], eps=1e-9)
    global_step = 0
    initial_epoch = 0
    if config['preload']:  
        model_filename = get_weights_file_path(config)
        print("preloading model")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        model.load_state_dict(state['model_state_dict'])

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for index, batch in enumerate(batch_iterator):
            model.train()
            encoder_input = batch['encoder_input'].to(device) # (batch_dim, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_dim, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_dim, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_dim, 1, seq_len, seq_len)
            label = batch['label'].to(device) # (batch_dim, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_dim, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_dim, seq_len, d_model)
            projection_output = model.project(decoder_output) # (batch_dim, seq_len, vocab_size)

            # (batch_dim, seq_len, vocab_size) -> (batch_dim, seq_len, vocab_size)
            loss = loss_fn(projection_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step=global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if index != 0 and index % 100 == 0:
                run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer, num_examples=1)

            global_step += 1

        # Save the model
        model_filename = get_weights_file_path(config)
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict()
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)