import torch
import sys
# from model4 import InputEmbedding, PositionalEncoding, MultiHeadAttentionBlock, FeedForwardBlock, EncoderBlock, DecoderBlock, Encoder, Decoder, ProjectionLayer
from transformer3 import Transformer, build_transformer, decode_text, word_to_idx, idx_to_word, causal_mask, vocab_size, greedy_decode

# Load the trained model
def load_model(model_path, device):
    model = build_transformer(vocab_size=vocab_size, seq_len=200, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()
    return model

# Preprocess input text
def text_to_tensor(text, seq_len, word_to_idx, device):
    tokens = text.lower().split()  # Basic tokenization
    token_ids = [word_to_idx.get(token, word_to_idx["<UNK>"]) for token in tokens]
    token_ids = [word_to_idx["<START>"]] + token_ids + [word_to_idx["<END>"]]
    
    # Pad or trim the sequence to seq_len
    if len(token_ids) < seq_len:
        token_ids.extend([word_to_idx["<PAD>"]] * (seq_len - len(token_ids)))
    else:
        token_ids = token_ids[:seq_len]

    return torch.tensor(token_ids).unsqueeze(0).to(device)  # Shape: (1, seq_len)

# Perform prediction
def predict(text, model, max_len, device):
    src_tensor = text_to_tensor(text, max_len, word_to_idx, device)
    src_mask = torch.ones((1, 1, max_len)).to(device)  # Adjust if necessary for the masking strategy used

    with torch.no_grad():
        output_ids = greedy_decode(model, src_tensor, src_mask, max_len, device)
    return decode_text(output_ids.cpu().numpy())

# Main prediction script
if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/transformer_model.pt"
    model = load_model(model_path, device)

    # Get user input
    print("Enter a sentence to generate a response (or type 'quit' to exit):")
    while True:
        input_text = input("Input: ")
        if input_text.lower() == 'quit':
            break

        # Generate prediction
        response = predict(input_text, model, max_len=200, device=device)
        print("Output:", response)
