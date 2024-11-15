import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ChatDataset(Dataset):

    def __init__(self, ds, tokenizer, seq_len):
        super(ChatDataset,self).__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        ds = self.ds[index]
        src = ds['source']
        tgt = ds['response']
        if len(src) > 190:
            src = src[:190]
        if len(tgt) > 190:
            tgt = tgt[:190]
        enc_input_tokens = self.tokenizer.encode(src).ids
        dec_input_tokens = self.tokenizer.encode(tgt).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sequence length is too long')

        # Label is the expected output from the decoder
        enc_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ])
        dec_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        assert enc_input.size(0) == self.seq_len
        assert dec_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': enc_input, # (seq_len)
            'decoder_input': dec_input, # (seq_len)
            'encoder_mask': (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            'decoder_mask': (dec_input != self.pad_token).unsqueeze(0).int() & causal_mask(dec_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,
            "src_text": src,
            "tgt_text": tgt,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0