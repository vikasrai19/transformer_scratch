from torch.utils.data import ConcatDataset

from data_model.cornell_dataset_model import conversation_data
from data_model.squad_dataset_model import qa_data

from collections import defaultdict
from nltk.tokenize import word_tokenize

# Step 1: Build vocabularies separately for each dataset
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

# Get vocabularies for each dataset
vocab_conversation = build_vocab(conversation_data)  # For Cornell
vocab_qa = build_vocab(qa_data)                      # For SQuAD

# Step 2: Combine vocabularies
combined_vocab = defaultdict(lambda: len(combined_vocab), vocab_conversation)  # Start with Cornell vocab

for word in vocab_qa:
    _ = combined_vocab[word]  # This ensures unique words are added only once

# Step 3: Convert defaultdict to a regular dictionary for easy access
combined_vocab = dict(combined_vocab)

print("Total vocabulary size:", len(combined_vocab))
print("Sample of combined vocabulary:", list(combined_vocab.items())[:10])



combined_data = ConcatDataset([conversation_data, qa_data])
