import torch
import torch.nn as nn
import numpy as np
import python_speech_features as psf
import librosa
from matplotlib import pyplot as plt
class LSTM_BN(nn.Module):
    def __init__(self, config, hidden_dim=256, win_length=12, num_lstm_layer=3):
        super(LSTM_BN, self).__init__()
        # here I'm going to assume the input is gonna be shaped as
        self.hidden_dim = hidden_dim
        self.win_length = win_length
        self.num_lstm_layer = num_lstm_layer
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(65 * (self.win_length * 2 + 1), self.hidden_dim, num_lstm_layer, batch_first=True)
        self.output_mat1 = nn.Linear(self.hidden_dim, 2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.sigmoid = nn.Sigmoid()
    def concate_frames(self, input_audio):
        padding = torch.zeros((input_audio.shape[0], self.win_length, input_audio.shape[2]))
        padded_input_audio = torch.cat([padding, input_audio, padding], dim=1)
        window_audio = []
        for i in range(0, input_audio.shape[1]):
            window_count = i + 12
            current_window = padded_input_audio[:, window_count-12:window_count+13]
            s = current_window.shape
            current_window = current_window.view((s[0], s[1] * s[2]))
            current_window = torch.unsqueeze(current_window, 1)
            window_audio.append(current_window)
        rtv = torch.cat(window_audio, dim=1)
        return rtv
    def forward(self, input_audio):
        mod_audio = self.concate_frames(input_audio)
        # here I'm assuming that the input_audio is of proper shape
        hidden_state = [torch.zeros((self.num_lstm_layer, mod_audio.shape[0], self.hidden_dim)), torch.zeros((self.num_lstm_layer, mod_audio.shape[0], self.hidden_dim))]
        out, hidden_state = self.lstm(mod_audio, hidden_state)
        # bn
        # x = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu(out)
        x = self.output_mat1(x)
        return x

    def test_forward(self, input_audio):
        x = self.forward(input_audio)
        return self.sigmoid(x)

class Gaze_aversion_detector():
    def __init__(self, config):
        if torch.cuda.is_available():
            dev = "cuda:1"
        else:
            dev = "cpu"
        self.model = LSTM_BN(config)
        self.config = config

    def __call__(self, sound_arr):
        # audio should be the
        sound_arr = (sound_arr - sound_arr.mean()) / sound_arr.std()
        sr = self.config["sample_rate"]
        appended = False
        if sound_arr.shape[0] < sr * 0.2:
            sound_arr = np.concatenate([np.zeros((int(sr * 0.2), )), sound_arr], axis=0)
            appended = True
        winstep = int(sr * self.config["window_length"])
        mfcc_feat = psf.mfcc(sound_arr, samplerate=sr, winlen=self.config["window_length"], nfft=2 * winstep, numcep=13)
        logfbank_feat = psf.logfbank(sound_arr, samplerate=sr, winlen=0.02, nfft=2 * winstep, nfilt=26)
        ssc_feat = psf.ssc(sound_arr, samplerate=sr, winlen=0.02, nfft=2 * winstep, nfilt=26)
        full_feat = np.concatenate([mfcc_feat, logfbank_feat, ssc_feat], axis=1)
        input_vec = torch.tensor(np.expand_dims(full_feat, axis=0), dtype=torch.float32)
        out_vec = self.model.test_forward(input_vec)[0]
        return out_vec