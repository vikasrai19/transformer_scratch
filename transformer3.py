import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.benchmark.utils.compare import optional_min
from tqdm import tqdm
import os
from model4 import InputEmbedding, PositionalEncoding, MultiHeadAttentionBlock, FeedForwardBlock, EncoderBlock, DecoderBlock, Encoder, Decoder, ProjectionLayer, init_weights
from data_model.dataset_model import word_to_idx, idx_to_word, vocab_size, dataloader, causal_mask, val_dataloader


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, embed: InputEmbedding, pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed = embed
        self.pos = pos
        self.projection_layer = projection_layer

    def encode(self, src, mask):
        src = self.embed(src)
        src = self.pos(src)
        return self.encoder(src, mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor, tgt: torch.Tensor):
        tgt = self.embed(tgt)
        tgt = self.pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(vocab_size: int, seq_len: int, d_model: int = 512, N: int  = 6, h: int = 8, dropout: float=0.1, d_ff: int = 2048, device = None) -> Transformer:
    embed = InputEmbedding(d_model, vocab_size)
    pos = PositionalEncoding(d_model, vocab_size, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, vocab_size)

    transformer = Transformer(encoder, decoder, embed, pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device : ", device)
model = build_transformer(vocab_size=vocab_size, seq_len=200, device=device)

def greedy_decode(model : Transformer, source, mask, max_len, device):
    # sos_idx = tokenizer.token_to_id("[SOS]")
    # eos_idx = tokenizer.token_to_id("[EOS]")
    sos_idx = word_to_idx["<START>"]
    eos_idx = word_to_idx["<END>"]

    print("model encode started")
    encoder_output = model.encode(source, mask)
    print("model encode completed")
    """
    Prepare a decoder input with the starting token
    """
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(mask).to(device)
        out = model.decode(encoder_output, mask, decoder_mask, decoder_input)
        print("model out value ", out)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        ).to(device)

        if next_word == eos_idx:
            break
    print("decoding completed and received decoder input")
    return decoder_input.squeeze(0)

def decode_text(data):
    words = []
    for d in data:
        words.append(idx_to_word[d])
    return " ".join(words)

def run_validation(model: Transformer, validation_ds, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    print("validation started")
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
            print("starting greedy decode")
            model_out = greedy_decode(model, encoder_input, encoder_mask, max_len, device)
            print("len of source text ", len(batch['src_text']))
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            # model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())
            model_out_text = decode_text(model_out.detach().cpu().numpy())
            print("received out text from model")

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


def train_model(model, dataloader, num_epochs=10, learning_rate=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<PAD>'], label_smoothing=0.1).to(device)
    initial_epoch = 0

    model_file_path = "./models/transformer_model.pt"
    if os.path.isfile(model_file_path):
        saved_model = torch.load(model_file_path, map_location=device)
        initial_epoch = saved_model['epoch'] + 1
        model.load_state_dict(saved_model['model_state_dict'])
        optimizer.load_state_dict(saved_model['optimizer_state_dict'])

    model.to(device)
    print("model is on device ", next(model.parameters()).device)

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

    for epoch in range(initial_epoch, num_epochs):
        model.train()
        total_loss = 0
        src_mask = None
        tgt_mask = None
        batch_iterator = tqdm(dataloader, desc=f'Epoch {epoch + 1} / {num_epochs}')
        for batch in batch_iterator:
            src = batch['encoder_input']
            tgt = batch['decoder_input']
            label = batch['label']
            src_mask = batch['encoder_mask']
            tgt_mask = batch['decoder_mask']
            enc_dec_mask = batch['encoder_decoder_mask']

            src = src.to(device)
            tgt = tgt.to(device)
            label = label.to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            enc_dec_mask = enc_dec_mask.to(device)

            encoder_output = model.encode(src, src_mask)
            decoder_output = model.decode(encoder_output, src_mask, tgt_mask, tgt)
            output = model.project(decoder_output)
            # loss = criterion(output.view(-1, vocab_size), tgt[:, :].reshape(-1))
            loss = criterion(output.view(-1, vocab_size), label.view(-1))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'models/transformer_model.pt')

        try:
            run_validation(model, val_dataloader, 200, device, lambda msg: batch_iterator.write(msg), None, None, num_examples=4)
        except Exception as e:
            print("exception while running validation ", e)

if __name__ == '__main__':
    train_model(model, dataloader=dataloader)