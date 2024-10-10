import random
import json
from model import build_transformer, Transformer
from data_model.chat_data_model2 import ChatDataModel, causal_mask
from config import get_config, get_weights_file, latest_weights_file_path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, random_split

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model : Transformer, source, mask, tokenizer: Tokenizer, max_len, device):
    # print("model source ", tokenizer.decode(source))
    sos_idx = tokenizer.token_to_id("[SOS]")
    eos_idx = tokenizer.token_to_id("[EOS]")
    
    encoder_output = model.encode(source, mask)

    """
    Prepare a decoder input with the starting token
    """
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(mask).to(device)
        out = model.decode(encoder_output, mask, decoder_mask, decoder_input)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)


def run_validation(model: Transformer, validation_ds, tokenizer: Tokenizer, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 60

    with torch.no_grad():
        for batch in validation_ds:
            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)
            print("len of source text ", len(batch['src_text']))
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())


            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
            count += 1

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def get_all_sentences(ds):
    all_sent = []
    for item in ds:
        all_sent.append(item)
    for sent in all_sent:
        yield sent


def get_or_build_tokenizer(config, ds):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    with open("./datasets/chat4.json", "r") as fl:
        ds_raw = json.load(fl)
    tokenizer = get_or_build_tokenizer(config, ds_raw)
    # ds_raw = ds_raw
    # ds_raw = ds_raw[:2500]
    # Keep 90% for training, 10% for validation
    print(f'Length of dataset: {len(ds_raw)}')
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = ChatDataModel(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = ChatDataModel(val_ds_raw, tokenizer, config['seq_len'])

    max_len = 0
    for idx in range(len(ds_raw) - 1):
        src_ids = tokenizer.encode(ds_raw[idx]).ids
        tgt_ids = tokenizer.encode(ds_raw[idx + 1]).ids
        max_len = max(max_len, len(src_ids))
        max_len = max(max_len, len(tgt_ids))

    print(f'Max length of sentence: {max_len}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=False)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer


def get_model(config, vocab_size, device=None):
    model = build_transformer(vocab_size, config['seq_len'], config['d_model'], device=device)
    return model


def train_model(config):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using Device : {device}")

    device = torch.device(device)
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = get_model(config, tokenizer.get_vocab_size(), device=device).to(device)
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file(config, preload) if preload else None

    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.mps.empty_cache()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_mask, decoder_input)
            project_output = model.project(decoder_output)

            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(project_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        model_filename = get_weights_file(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

        try:
            run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer, num_examples=4)
        except Exception as e:
            print("exception while running validation ", e)
            pass


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
