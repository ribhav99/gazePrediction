import sys
sys.path.append('..')
import torch.nn as nn
import torch

import utils # type: ignore

def export_config():
    return {
        "conv_layers": [1, 8, 32, 64, 128, 256],
        "kernel_size": 5,
        "padding": 2,
        "activation_fn": nn.ReLU(),
        "pool": nn.AvgPool2d(2),
        # "pool": None,
        "epochs": 100,
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        # "loss_fn": nn.MultiLabelSoftMarginLoss(),
        # "loss_fn": nn.BCELoss(),
        "loss_fn": utils.weighted_binary_cross_entropy,
        "learning_rate": 0.00001,
        "batch_size": 64,
        "wandb": False,
        "load_model": False,
        "early_stopping": 5,
        "sample_length": 5,
        "window_length": 0.01,
        "time_step": 0.01, # window_length / this shoulkd be <= 1201 (for 5 sec samples)
        "use_listener": False
    }
def export_config_Evan():
    return {
        "kernel_size": 5,
        "padding": 2,
        "activation_fn": nn.ReLU(),
        "epochs": 100,
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',

        # "loss_fn": nn.MultiLabelSoftMarginLoss(),
        "loss_fn": nn.BCELoss(),
        # "loss_fn": utils.weighted_binary_cross_entropy,

        "model_type": "BasicTransformer",
        "input_modality": "intensity, sentence_structure",
        "learning_rate": 0.0001,
        "batch_size": 512,
        "wandb": False,
        "load_model": False,
        "model_id": "gzqjeoud",
        "model_name": "time=2023-01-07 20:09:13.585099_epoch=99.pt",
        "early_stopping": 5,
        "sample_length": 5,
        "window_length": 0.01,
        "time_step": 0.01, # window_length / this shoulkd be <= 1201 (for 5 sec samples)
        "use_listener": True,
        "sample_rate": 22000
    }