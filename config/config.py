import torch.nn as nn
import torch

def export_config():
    return {
        "conv_layers": [1, 4, 8],
        "kernel_size": 5,
        "activation_fn": nn.ReLU(),
        "epochs": 100,
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "loss_fn": nn.MultiLabelSoftMarginLoss(),
        "learning_rate": 0.001,
        "batch_size": 64,
        "wandb": True,
        "load_model": False,
        "early_stopping": 5
    }