import torch
from torch import nn
import math

# Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# Feed-Forward Block
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# Input Embedding
class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float, device) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = device

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pos_enc', pe)

    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1), :].to(self.device)
        return self.dropout(x)


# Residual Connection
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.norm(x)))


# Multi-Head Attention Block
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask):
        d_k = query.size(-1)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = scores.softmax(dim=-1)
        if self.dropout:
            scores = self.dropout(scores)
        return scores @ value

    def forward(self, q, k, v, mask):
        query = self.w_q(q).view(q.size(0), q.size(1), self.h, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(k.size(0), k.size(1), self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(v.size(0), v.size(1), self.h, self.d_k).transpose(1, 2)
        x = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(q.size(0), -1, self.h * self.d_k)
        return self.w_o(x)


# Encoder and Decoder Blocks
class EncoderBlock(nn.Module):
    def __init__(self, d_model, self_attn, ff_block, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.ff_block = ff_block
        self.residual = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.residual[1](x, self.ff_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(layers[0].self_attn.w_q.out_features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, ff_block, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff_block = ff_block
        self.residual = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.residual[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.residual[1](x, lambda x: self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.residual[2](x, self.ff_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(layers[0].self_attn.w_q.out_features)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)


# Domain-Specific Projection Layers
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)
