import os
from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 30,
        "lr": 10**-4,
        "seq_len": 200,
        "d_model": 512,
        "model_folder": "weights",
        # "model_basename": "chat_model_1",
        "model_basename": "chat_model_2",
        "preload":"latest",
        # "tokenizer_file": "tokenizer.json",
        "tokenizer_file": "tokenizer2.json",
        "experiment_name": "runs/chat_model"
    }

def get_weights_file(config, epoch):
    model_folder = f'{config["model_folder"]}'
    # model_filename = f'{config["model_basename"]}{epoch}.pt'
    model_filename = f'{config["model_basename"]}.pt'
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f'{config["model_folder"]}'
    model_filename = f'{config["model_basename"]}*'
    weight_files = list(Path(model_folder).glob(model_filename))
    if len(weight_files) == 0:
        return None
    
    weight_files.sort()
    return str(weight_files[-1])