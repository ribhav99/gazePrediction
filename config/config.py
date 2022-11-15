import torch.nn as nn

def export_config():
    return {
        "conv_layers": [1, 4, 8],
        "kernel_size": 5,
        "activation_fn": nn.ReLU()
    }