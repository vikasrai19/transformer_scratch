import os
import torch
import torch.nn as nn
from model2 import Encoder, Decoder, ProjectionLayer
from data_model.dataset_model import dataloader, word_to_idx, vocab_size, idx_to_word

import torch.optim as optim
from tqdm import tqdm

class Transfomer(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super(Transfomer, self).__init__()
        self.encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
        self.decoder = Decoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
        self.projection = ProjectionLayer(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, enc_dec_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, enc_dec_mask)
        output = self.projection(dec_output)
        return output

if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.mps.empty_cache()
else:
    device = torch.device("cpu")
print("device ", device)

def prepare_input(sentence, max_len=200):
    indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in sentence.split()]
    padded_indeces = indices + [word_to_idx["<PAD>"]] * (max_len - len(indices))
    return torch.tensor(padded_indeces).unsqueeze(0) # Add batch dimension

def create_mask(seq):
    # Create a mask where padding tokens are ignored (0s for actual tokens, 1s for padding)
    return (seq != word_to_idx["<PAD>"]).unsqueeze(1).unsqueeze(2)  # Shape (batch_size, 1, 1, seq_len)


def generate_output(model, input_sentence, max_len, src_mask, tgt_mask):
    # Prepare the input
    src = prepare_input(input_sentence, max_len).to(device)
    # src_mask = create_mask(src).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Prepare the target sequence (initially just the start token)
    tgt = [word_to_idx["<START>"]]  # Start with the padding token (or a special start token if defined)
    for _ in range(max_len):  # Limit the output length
        tgt_tensor = torch.tensor(tgt).unsqueeze(0).to(device)  # Add batch dimension
        # tgt_mask = create_mask(tgt_tensor).to(device)

        # Generate decoder input by excluding the last token for teacher forcing
        output = model(src, tgt_tensor, src_mask, tgt_mask)

        # Get the predicted token (argmax)
        next_token = output[:, -1, :].argmax(dim=-1).item()  # Get the last predicted token
        tgt.append(next_token)  # Append predicted token to the target sequence

        # Stop if the predicted token is the end token (if defined)
        if next_token == word_to_idx["<PAD>"] or next_token == word_to_idx['<END>']:  # Change to end token if defined
            break

    # Convert indices back to words
    output_sentence = ' '.join([idx_to_word[idx] for idx in tgt if idx in idx_to_word])
    return output_sentence


def train_model(model, dataloader, num_epochs=10, learning_rate=1e-7):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<PAD>'], label_smoothing=0.1).to(device)
    initial_epoch = 0

    model_file_path = "./models/transformer_model.pt"
    if os.path.isfile(model_file_path):
        saved_model = torch.load(model_file_path)
        initial_epoch = saved_model['epoch'] + 1
        model.load_state_dict(saved_model['model_state_dict'])
        optimizer.load_state_dict(saved_model['optimizer_state_dict'])


    for epoch in range(initial_epoch, num_epochs):
        model.train()
        total_loss = 0
        src_mask = None
        tgt_mask = None
        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1} / {num_epochs} - total loss : {total_loss}'):
            src = batch['encoder_input']
            tgt = batch['decoder_input']
            src_mask = batch['encoder_mask']
            tgt_mask = batch['decoder_mask']
            enc_dec_mask = batch['encoder_decoder_mask']
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            enc_dec_mask = enc_dec_mask.to(device)

            assert torch.all(torch.isfinite(src)), "Input contains NaNs or Infs"
            assert torch.all(torch.isfinite(tgt)), "Target contains NaNs or Infs"


            # src_mask = None
            # tgt_mask = None
            # enc_dec_mask = None
            output = model(src, tgt[:, :], src_mask, tgt_mask, enc_dec_mask)
            print("ouput ", output)
            loss = criterion(output.view(-1, vocab_size), tgt[:, :].reshape(-1))
            total_loss += loss.item()
            print("loss item ", loss)

            optimizer.zero_grad()
            loss.backward()
            print("loss item after backward pass ", loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        print("total loss ", total_loss)
        print("dataloader length ", len(dataloader))
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'models/transformer_model.pt')

        input_sentence = [
            "hello",
            "How are you doing",
            "I like programming very much",
            "i thin i have a crush on you"
        ]
        for inp in input_sentence:
            output = generate_output(model, inp, max_len=200, src_mask=src_mask, tgt_mask=tgt_mask)
            print(f"Input: {inp} - Output: {output}")

model = Transfomer(vocab_size, d_model=128, num_heads=8, d_ff=256, num_layers=6, max_len=200)
model.to(device)
train_model(model, dataloader)