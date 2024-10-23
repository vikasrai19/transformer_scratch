import os
import wget
import zipfile
import nltk
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict
from nltk.tokenize import word_tokenize
import time

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
    conversations = []
    with open(file_path, 'r', encoding='iso-8859-1') as f:
        for line in f.readlines():
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 5:
                conversations.append((parts[3], parts[4]))  # Extract conversation pair
    return conversations

# Vocabulary Building and Preprocessing
def build_vocab(data):
    vocab = defaultdict(lambda: len(vocab))
    vocab["<pad>"] = 0
    vocab["<sos>"] = 1
    vocab["<eos>"] = 2
    vocab["<unk>"] = 3

    for pair in data:
        for sentence in pair:
            for word in word_tokenize(sentence.lower()):
                vocab[word]  # Add word to vocab
    return vocab

# Encode conversation pairs
def encode_pair(pair, vocab):
    src, tgt = pair
    src_tokens = [vocab["<sos>"]] + [vocab[word] for word in word_tokenize(src.lower())] + [vocab["<eos>"]]
    tgt_tokens = [vocab["<sos>"]] + [vocab[word] for word in word_tokenize(tgt.lower())] + [vocab["<eos>"]]
    return src_tokens, tgt_tokens

# Pad sequence
def pad_sequence(seq, max_length, pad_value=0):
    return seq[:max_length] + [pad_value] * max(0, max_length - len(seq))

# Cornell Dataset Class
class CornellDataset(Dataset):
    def __init__(self, conversations, vocab, max_length=30):
        self.data = [encode_pair(pair, vocab) for pair in conversations]
        self.max_length = max_length
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src = pad_sequence(src, self.max_length)
        tgt = pad_sequence(tgt, self.max_length)
        # return torch.tensor(src), torch.tensor(tgt), torch.tensor(0)
        return src, tgt

# Download and load the conversation dataset
download_cornell_dataset()
conversation_data = load_conversation_dataset()
vocab_conversation = build_vocab(conversation_data)
cornell_dataset = CornellDataset(conversation_data, vocab_conversation)

# Create DataLoader for conversation data
dataset_len = len(cornell_dataset)
print("len of conv dataset ", dataset_len)
portion_size = 1 * dataset_len
subset_dataset = Subset(cornell_dataset, list(range(int(portion_size))))
conversation_loader = DataLoader(subset_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=0)
