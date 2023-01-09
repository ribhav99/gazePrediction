import torch
import torch.nn as nn

class BasicTransformerModel(nn.Module):
    def __init__(self, config, input_dim=65, hidden_dim=64):
        super(BasicTransformerModel, self).__init__()
        # here I'm going to assume the input is gonna be shaped as
        self.hidden_dim = hidden_dim
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.transformer = nn.Transformer(d_model=self.hidden_dim, batch_first=True)
        self.output_mat1 = nn.Linear(self.hidden_dim, 1)
        self.relu = nn.ReLU()
        self.config = config
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_audio, target):
        # here I'm assuming that the input_audio is of proper shape
        out = self.transformer(input_audio, target)
        print(out.shape)
        # bn
        # x = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu(out)
        x = self.output_mat1(x)
        x = x[:, :, 0]
        x = self.sigmoid(x)
        return x

    def test_forward(self, input_audio):
        x = self.forward(input_audio)
