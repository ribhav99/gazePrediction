import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import export_config # type: ignore

class CNNet(nn.Module):
    def __init__(self, config, single_datapoint_shape=(1,40,1201), target_shape=50):
        super().__init__()
        self.conv_layers = []
        conv_layers = config['conv_layers']
        kernel_size = config['kernel_size']
        for i in range(len(conv_layers)):
            if i < len(conv_layers) - 1:
                in_ = conv_layers[i]
                out_ = conv_layers[i+1]
                self.conv_layers.append(nn.Conv2d(in_, out_, kernel_size=kernel_size))
                self.conv_layers.append(nn.MaxPool2d(2))
                self.conv_layers.append(nn.BatchNorm2d(out_))
                self.conv_layers.append(config['activation_fn'])

        self.conv = nn.Sequential(*self.conv_layers)

        fake_data = torch.ones(single_datapoint_shape)
        fake_data = fake_data.unsqueeze(1)
        fake_data = self.conv(fake_data)

        fc_input = torch.prod(torch.tensor(fake_data.shape))
        self.fc1 = nn.Linear(fc_input, fc_input//4)
        self.fc2 = nn.Linear(fc_input//4, target_shape)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.sigmoid(x)
    
    def load_weights(self, pretrained_dict):
    #   not_copy = set(['fc.weight', 'fc.bias'])
        not_copy = set()
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in not_copy}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

if __name__ == '__main__':
    from createDataset import AudioDataset

    wav_5_sec_dir = '../data/wav_files_5_seconds/'
    gaze_dir = '../data/gaze_files'
    config = export_config()
    print('Initialising Dataset')

    dataset = AudioDataset(wav_5_sec_dir, gaze_dir, 5, 0.1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    model = CNNet(config)

    for batch, (x, y) in enumerate(dataloader):
        print(x.shape, y.shape)
        print(model(x))
        break