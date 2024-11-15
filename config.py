from pathlib import Path

def get_config(): 
    return {
        'batch_size': 8,
        "num_epochs": 5,
        "lr": 1e-7,
        "seq_len"  : 200,
        "d_model": 128,
        "model_foldername": "weights",
        "model_name": "transfomer_model2.pt",
        "preload": None,
        "tokenizer_file": "tokenizer.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config):
    model_folder = config['model_foldername']
    model_name = config['model_name']
    return str(Path('.') / model_folder / model_name)