import torch.nn as nn
import torch

def export_config():
    return {
        "conv_layers": [1, 8, 32, 64, 128, 256],
        "kernel_size": 5,
        "padding": 2,
        "activation_fn": nn.ReLU(),
        "pool": nn.MaxPool2d(2),
        # "pool": None,
        "epochs": 100,
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "loss_fn": nn.MultiLabelSoftMarginLoss(),
        "learning_rate": 0.00001,
        "batch_size": 64,
        "wandb": True,
        "load_model": False,
        "early_stopping": 5,
        "window_length": 0.1
    }