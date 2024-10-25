import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from model import InputEmbedding, PositionalEncoding, ProjectionLayer, EncoderBlock, DecoderBlock, MultiHeadAttentionBlock, FeedForwardBlock, Encoder, Decoder
from data_model.combined_data_model import combined_data, combined_vocab

# Shared Transformer Model with Domain Heads
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, embed, pos, projection_heads):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed = embed
        self.pos = pos
        self.projection_heads = nn.ModuleDict(projection_heads)

    def encode(self, src, mask):
        src = self.pos(self.embed(src))
        return self.encoder(src, mask)

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        tgt = self.pos(self.embed(tgt))
        return self.decoder(tgt, enc_out, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask, domain):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.projection_heads[domain](dec_out)


# Example Initialization
def create_transformer(d_model, vocab_size, num_layers, d_ff, num_heads, dropout, domains, device):
    # Instantiate layers
    embed = InputEmbedding(d_model, vocab_size)
    pos = PositionalEncoding(d_model, 512, dropout, device)
    projection_heads = {domain: ProjectionLayer(d_model, vocab_size) for domain in domains}
    
    # Encoder and Decoder layers
    encoder_layers = nn.ModuleList([
        EncoderBlock(d_model, MultiHeadAttentionBlock(d_model, num_heads, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout)
        for _ in range(num_layers)
    ])
    decoder_layers = nn.ModuleList([
        DecoderBlock(d_model, MultiHeadAttentionBlock(d_model, num_heads, dropout), MultiHeadAttentionBlock(d_model, num_heads, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout)
        for _ in range(num_layers)
    ])
    
    # Instantiate encoder, decoder, and transformer
    encoder = Encoder(encoder_layers)
    decoder = Decoder(decoder_layers)
    transformer = Transformer(encoder, decoder, embed, pos, projection_heads)
    
    return transformer.to(device)


def custom_collate_fn(batch):
    src_seqs, tgt_seqs, domains = zip(*batch)
    
    # Stack sequences and domain labels
    src_seqs = torch.stack(src_seqs)
    tgt_seqs = torch.stack(tgt_seqs)
    domains = torch.tensor(domains)  # 0 for Cornell, 1 for SQuAD
    
    return src_seqs, tgt_seqs, domains


batch_size = 32
combined_loader = DataLoader(
    combined_data, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=custom_collate_fn
)

def train_transformer_multi_domain(transformer, combined_loader, num_epochs, learning_rate, device):
    optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)
    transformer.train()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        epoch_loss = 0.0
        for src, tgt, domain in tqdm(combined_loader, desc="Training on combined dataset"):
            src, tgt, domain = src.to(device), tgt.to(device), domain.to(device)
            
            # Create masks for source and target sequences
            src_mask = (src != 0).unsqueeze(-2)
            tgt_mask = (tgt != 0).unsqueeze(-2)
            
            # Shift target sequence for teacher forcing
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass with domain-specific head
            output = transformer(src, tgt_input, src_mask, tgt_mask, domain=domain)
            
            # Compute loss
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)), 
                tgt_output.view(-1), 
                ignore_index=0  # Assuming 0 is padding token
            )
            
            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(combined_loader)
        print(f"Average loss: {avg_loss:.4f}")

    print("Training complete.")

# Training parameters
vocab_size = len(combined_vocab)
num_epochs = 10
learning_rate = 1e-4
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
transformer = create_transformer(d_model=512, vocab_size=vocab_size, num_layers=6, d_ff=2048, num_heads=8, dropout=0.1, domains=['general', 'cars'], device=device)


# Train the model
train_transformer_multi_domain(transformer, combined_loader, num_epochs, learning_rate, device)
