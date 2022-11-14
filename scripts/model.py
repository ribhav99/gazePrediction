import torch
import torch.nn as nn

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
                self.conv_layers.append(nn.ReLU())

        self.conv = nn.Sequential(*self.conv_layers)

        fake_data = torch.ones(single_datapoint_shape)
        print(fake_data.shape)
        fake_data = self.conv(fake_data)

        self.fc1 = nn.Linear(torch.prod(fake_data.shape), torch.prod(fake_data.shape)//4)
        self.fc2 = nn.Linear(torch.prod(fake_data.shape)//4, target_shape)


    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x.flatten())
        x = self.fc2(x)
        return nn.Sigmoid(x)

if __name__ == '__main__':
    from createDataset import AudioDataset

    wav_5_sec_dir = '../data/wav_files_5_seconds/'
    gaze_dir = '../data/gaze_files'
    config = {"conv_layers": [1, 4, 8, 16, 32, 64], "kernel_size": 5}
    print('Initialising Dataset')

    dataset = AudioDataset(wav_5_sec_dir, gaze_dir, 5, 0.1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    model = CNNet(config)

    for batch, (x, y) in enumerate(dataloader):
        print(x.shape, y.shape)
        print(model(x))
        break