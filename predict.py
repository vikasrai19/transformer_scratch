import json
from pathlib import Path
from config import get_config, latest_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer
from data_model.chat_data_model import ChatDataModel, causal_mask
import torch
import sys


def predict(sentence):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using Device : ", device)
    device = torch.device(device)

    config = get_config()
    tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_file'])))

    model = build_transformer(tokenizer.get_vocab_size(), config['seq_len'], config['d_model'], device=device).to(device)

    model_file_name = latest_weights_file_path(config)
    state = torch.load(model_file_name)
    model.load_state_dict(state['model_state_dict'])
    label = ""

    if type(sentence) == int or sentence.isdigit():
        id = int(sentence)
        with open("./datasets/chat3.json", "r") as fl:
            ds_raw = json.load(fl)['messages']
        
        ds = ChatDataModel(ds_raw, tokenizer, config['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]["tgt_text"]
    seq_len = config['seq_len']

    model.eval()
    with torch.no_grad():
        source = tokenizer.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64),
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).unsqueeze(0).to(device)
        print("source shape ", source.shape)
        source_mask = (source != tokenizer.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).int().to(device)
        encoder_output = model.encode(source, source_mask)
        decoder_input = torch.empty(1, 1).fill_(tokenizer.token_to_id('[SOS]')).type_as(source).to(device)

        # Print the source sentence and target start prompt
        if label != "": print(f"{f'ID: ':>12}{id}") 
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "": print(f"{f'TARGET: ':>12}{label}") 
        print(f"{f'PREDICTED: ':>12}", end='')

        decoder_input = torch.empty(1, 1).fill_(tokenizer.token_to_id('[SOS]')).type_as(source).to(device)
        while True:
            if decoder_input.size(1) == config['seq_len']:
                break

            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_mask, decoder_input)

            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
            )

            if next_word == tokenizer.token_to_id('[EOS]'):
                break

        # while decoder_input.size(1) < seq_len:
        #     decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int64).type_as(source).to(device)
        #     out = model.decode(encoder_output, source_mask, decoder_mask, decoder_input)
        #     prob = model.project(out[:, -1])
        #     _, next_word = torch.max(prob, dim=1)
        #     decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        #     # print the translated word
        #     print(f"{tokenizer.decode([next_word.item()])}", end=' ')

        #     # break if we predict the end of sentence token
        #     if next_word == tokenizer.token_to_id('[EOS]'):
        #         break
    return tokenizer.decode(decoder_input.squeeze(0).detach().cpu().numpy())

print(predict(sys.argv[1] if len(sys.argv) > 1 else "HI How are you doing"))