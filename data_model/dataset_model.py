import json
import nltk
import zipfile
import torch
import os
import wget
import time
from datasets import load_dataset
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

nltk.download('punkt')
nltk.download('punkt_tab')

# Download Cornell Movie Dialogs Corpus
def download_cornell_dataset(retries=10, delay=5):

    url = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    if not os.path.exists("cornell_movie_dialogs_corpus.zip"):
        for attempt in range(retries):
            try:
                wget.download(url, "cornell_movie_dialogs_corpus.zip")
                break
            except Exception as e:
                print(f"Error occurred: {e}")
                if attempt < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("Failed to download the dataset after several attempts.")
                    raise
    with zipfile.ZipFile("cornell_movie_dialogs_corpus.zip", 'r') as zip_ref:
        zip_ref.extractall("cornell_movie_data")

# Load and preprocess the conversation data
def load_conversation_dataset():
    file_path = "cornell_movie_data/cornell movie-dialogs corpus/movie_lines.txt"
    # file_path = "cornell_movie_data/cornell movie-dialogs corpus/movie_lines.txt"
    # conversations = []
    # with open(file_path, 'r', encoding='iso-8859-1') as f:
    #     for line in f.readlines():
    #         parts = line.strip().split(" +++$+++ ")
    #         if len(parts) == 5:
    #             conversations.append((parts[3], parts[4]))  # Extract conversation pair

    file_path = "datasets/chatbot_dataset.json"
    with open(file_path, "r") as file:
        data = json.load(file)
        print("data type ", type(data))
    final_conversation = []
    for i in range(len(data) - 1):
        if len(data[i]['source']) < 200 and len(data[i]['response']) < 200:
            final_conversation.append((data[i]['source'], data[i]['response']))
        # if len(conversations[i][1]) < 200 and len(conversations[i + 1][1]) < 200:
        #     final_conversation.append((conversations[i][1], conversations[i + 1][1]))

    return final_conversation


conv = load_conversation_dataset()
print("conv len ", len(conv))

def build_vocab(data, min_freq=2):
    words = []
    for src, _ in data:
        words.extend(src.split())

    word_counts = Counter(words)
    vocab = {word: idx for idx, (word, count) in enumerate(word_counts.items()) if count >= min_freq }
    vocab['<PAD>'] = len(vocab)
    vocab['<UNK>'] = len(vocab)
    vocab['<START>'] = len(vocab)
    vocab['<END>'] = len(vocab)
    return vocab

word_to_idx = build_vocab(conv)
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(word_to_idx)


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

class TransformerDataLoader(Dataset):
    def __init__(self, data, word_to_idx, max_length=200):
        self.data = data
        self.word_to_idx = word_to_idx
        self.max_len = max_length

        self.sos_token = torch.tensor([word_to_idx['<START>']], dtype=torch.int64)
        self.pad_token = torch.tensor([word_to_idx['<PAD>']], dtype=torch.int64)
        self.eos_token = torch.tensor([word_to_idx['<END>']], dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def generate_indices(self, sentence):
        return [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in sentence.split()]

    def __getitem__(self, index):
        src, tgt = self.data[index]
        enc_input_tokens = self.generate_indices(src)
        dec_input_tokens = self.generate_indices(tgt)

        enc_num_padding_tokens = self.max_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.max_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ], dim=0)

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ], dim=0)

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ], dim=0)

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'encoder_decoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'label': label,
            'src_text': src,
            'tgt_text': tgt
        }
        # return encoder_input, decoder_input

conv_len = int(0.1 * len(conv))
final_conv = conv[:conv_len]

train_ds_size = int(0.8 * len(final_conv))
test_ds_size = len(final_conv) - train_ds_size
train_ds, val_ds = random_split(final_conv, [train_ds_size, test_ds_size])
dataset = TransformerDataLoader(train_ds, word_to_idx)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

val_dataset = TransformerDataLoader(val_ds, word_to_idx)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

print("dataset length ", conv_len)
print("training dataset length ", len(train_ds))