import os
import wget
import zipfile
import nltk
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict
from datasets import load_dataset
from data_model.cornell_dataset_model import pad_sequence, build_vocab
from nltk.tokenize import word_tokenize
import time

nltk.download('punkt')
nltk.download('punkt_tab')


def encode_pair(pair, vocab):
    src, tgt = pair[1], pair[2]
    src_tokens = [vocab["<sos>"]] + [vocab[word] for word in word_tokenize(src.lower())] + [vocab["<eos>"]]
    tgt_tokens = [vocab["<sos>"]] + [vocab[word] for word in word_tokenize(tgt.lower())] + [vocab["<eos>"]]
    return src_tokens, tgt_tokens


# Load SQuAD dataset
def load_squad_dataset():
    dataset = load_dataset("squad")
    qa_pairs = []
    
    for entry in dataset['train']:  # We use the training split for example
        context = entry['context']
        question = entry['question']
        answer = entry['answers']['text'][0]  # Take the first answer
        qa_pairs.append((context, question, answer))
    
    return qa_pairs

# Preprocess SQuAD dataset
class SQuADDataset(Dataset):
    def __init__(self, qa_pairs, vocab, max_length=30):
        self.data = [encode_pair(pair, vocab) for pair in qa_pairs]
        self.max_length = max_length
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src = pad_sequence(src, self.max_length)
        tgt = pad_sequence(tgt, self.max_length)
        domain = 1  # Set domain as 0 for Cornell dataset
        return torch.tensor(src), torch.tensor(tgt), domain

# Load the QA dataset
qa_data = load_squad_dataset()
vocab_qa = build_vocab(qa_data)