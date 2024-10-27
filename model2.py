import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len= 200):
        """
        d_model represents the number of features in the input sequence
        max_len represents the max words that can be present in the sentence

        Lets consider d_model = 512 and max_len = 200
        Then shape of pe = (200, 512) ie 200 rows [1 row is 1 word] and each word represented by 512 columns [dimensions]
        For each row we of pe, we need one value for positional encoding .. hence we generate positional encoding value from 1 to max len ie [1, 2, 3 .... max_len] .. the shape of this matrix will be (200,) .. unsqueeze(1) converts this 1d array into 2d array with 1 column max_len as rows
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model) # shape : (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # shape : (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()) * (-math.log(10000.0) / d_model) # shape : (d_model / 2, )
        pe[:, 0::2] = torch.sin(position * div_term) # shape : (max_len, d_model / 2)
        pe[:, 1::2] = torch.cos(position * div_term) # shape : (max_len, d_model / 2)
        pe = pe.unsqueeze(0) # shape : (1, max_len, d_model)
        # pe = pe.transpose(0, 1)
        # print("shape of pe ", pe.shape)
        self.register_buffer('pe', pe) # shape : (1, max_len, d_model)

    def forward(self, x):
        """
        x.size(0) -> presents the sequence length of the sentence .. i.e., the number of tokens in the current input batch
        The line x = x + self.pe[:x.size(0), :] adds positional encodings to the input embeddings. This ensures that each token embedding now contains both its original content information and its position in the sequence, enabling the model to better understand the relationships and order among tokens in the input data.
        """
        x = x + self.pe[:x.size(0), :x.size(1)]
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        d_model represents the number of features in the input sequence
        num_heads represents the number of attention heads
        """
        super(MultiHeadedAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False) # parameters - (input size, output size)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.dense = nn.Linear(d_model, d_model, bias=False) # Layer for add and norm

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        x = x.transpose(1, 2)
        return x

    def forward(self, q, k, v, mask=None):
        """
        shape of q, k and v -> (batch_size, sequence_length, d_model)
        """
        batch_size = q.shape[0]
        Q = self.wq(q)
        K = self.wk(k)
        V = self.wv(v)

        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.depth ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return output
        # batch_size = q.shape[0]
        # q = self.split_heads(self.wq(q), batch_size)
        # k = self.split_heads(self.wk(k), batch_size)
        # v = self.split_heads(self.wv(v), batch_size)
        #
        # scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.depth)
        # print("score shape ", scores.shape)
        # if mask is not None:
        #     print("mask shape ", mask.shape)
        #     scores = scores.masked_fill(mask == 0, -1e9)
        #     print("score shape 2 ", scores.shape)
        # attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        #
        # context = torch.matmul(attention_weights, v)
        # context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # output = self.dense(context)
        # return output


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardLayer, self).__init__()
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc_2(torch.nn.functional.relu(self.fc_1(x)))


class AddNormLayer(nn.Module):
    def __init__(self, d_model):
        super(AddNormLayer, self).__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadedAttention(d_model, num_heads)
        self.add_norm_1 = AddNormLayer(d_model)
        self.ffn = FeedForwardLayer(d_model, d_ff)
        self.add_norm_2 = AddNormLayer(d_model)

    def forward(self, x, mask=None):
        attn_output = self.mha(x, x, x, mask)
        out1 = self.add_norm_1(x, attn_output)

        ffn_output = self.ffn(out1)
        out2 = self.add_norm_2(out1, ffn_output)
        return out2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.masked_mha = MultiHeadedAttention(d_model, num_heads)
        self.add_norm_1 = AddNormLayer(d_model)

        self.encoder_decoder_attention = MultiHeadedAttention(d_model, num_heads)
        self.add_norm_2 = AddNormLayer(d_model)

        self.ffn = FeedForwardLayer(d_model, d_ff)
        self.add_norm_3 = AddNormLayer(d_model)

    def forward(self, x, enc_output, mask=None, enc_dec_mask=None):
        masked_attention_output = self.masked_mha(x, x, x, mask)
        x = self.add_norm_1(x, masked_attention_output)

        # Encoder decoder attention
        enc_dec_attention_output = self.encoder_decoder_attention(x, enc_output, enc_output, enc_dec_mask)
        x = self.add_norm_2(x, enc_dec_attention_output)

        ffn_output = self.ffn(x)
        x = self.add_norm_3(x, ffn_output)

        return x


class Encoder(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super(Encoder, self).__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff)  for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super(Decoder, self).__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, x, enc_output, mask=None, enc_dec_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, mask, enc_dec_mask)

        return x


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_sizes):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(d_model, vocab_sizes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


