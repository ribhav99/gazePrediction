import torch
import torch.nn as nn
import numpy as np
import python_speech_features as psf
import librosa
from matplotlib import pyplot as plt
from config.config import export_config_Evan
import utils
from sklearn import metrics
from Transfomer_models import *

def load_model(config):
    import wandb
    # initialize model
    if 'LSTM_BN' == config['model_type']:
        model = LSTM_BN(config)
    elif config["model_type"] == "BasicTransformer":
        model = BasicTransformerModel(config, input_dim=8)
    elif config['model_type'] == "LSTM_no_concat_frame":
        model = LSTM_BN_no_concate(config)
    elif config["model_type"] == "LSTM_no_concat_frame_intensity_sentence_structure_input" and config["use_listener"]:
        model = LSTM_small(config, input_dim=18)
    elif config["model_type"] == "LSTM_no_concat_frame_intensity_sentence_structure_simplified_input" and config["use_listener"]:
        model = LSTM_small(config, input_dim=8)
    elif config["model_type"] == "LSTM_frame_intensity_sentence_structure_simplified_input" and config["use_listener"]:
        model = LSTM_BN(config, input_dim=8)
    elif config["model_type"] == "LSTM_no_concat_frame_intensity_sentence_structure_input_larger":
        model = LSTM_small(config, input_dim=9, num_lstm_layer=3)
    elif config["model_type"] == "LSTM_no_concat_frame_intensity_sentence_structure_input":
        model = LSTM_small(config, input_dim=9)
    elif config["model_type"] == "LSTM_no_concat_frame_intensity_sentence_structure_simplified_input_bidirection" and config["use_listener"]:
        model = LSTM_small(config, input_dim=8, bidirection=True, hidden_dim=64, num_lstm_layer=2)
    print("training model: {} \n We use Listener Data: {}".format(config['model_type'], config["use_listener"]))        
    # load model weights
    if config["load_model"]:
        if config['wandb']:
            wandb.login()
            run_obj = wandb.init(project="gaze_prediction", config=config, save_code=True,
                resume='allow', id=config["model_id"])
            checkpoint_name = config["model_name"]
            wandb.restore(checkpoint_name,
                                    run_path='gaze_prediction_team/gaze_prediction/{}'.format(run_obj.id))
            checkpoint_path = utils.find_path(checkpoint_name, 'wandb')
            pretrained_dict = torch.load(checkpoint_path, map_location=config['device'])
            model.load_weights(pretrained_dict)
        else:
            wandb = None
    model.to(config['device'])
    return model

class LSTM_BN(nn.Module):
    def __init__(self, config, hidden_dim=64, win_length=12, num_lstm_layer=3, input_dim=65):
        super(LSTM_BN, self).__init__()
        # here I'm going to assume the input is gonna be shaped as
        self.hidden_dim = hidden_dim
        self.win_length = win_length
        self.num_lstm_layer = num_lstm_layer
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim * (self.win_length * 2 + 1), self.hidden_dim, num_lstm_layer, batch_first=True)
        self.output_mat1 = nn.Linear(self.hidden_dim, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.config = config
        self.sigmoid = nn.Sigmoid()
    def concate_frames(self, input_audio):
        padding = torch.zeros((input_audio.shape[0], self.win_length, input_audio.shape[2])).to(self.config["device"])
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
        hidden_state = [torch.zeros((self.num_lstm_layer, mod_audio.shape[0], self.hidden_dim)).to(self.config["device"]), 
        torch.zeros((self.num_lstm_layer, mod_audio.shape[0], self.hidden_dim)).to(self.config["device"])]
        out, hidden_state = self.lstm(mod_audio, hidden_state)
        # bn
        # x = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu(out)
        x = self.output_mat1(x)
        x = x[:, :, 0]
        x = self.sigmoid(x)
        return x

    def test_forward(self, input_audio):
        x = self.forward(input_audio)
class LSTM_BN_no_concate(nn.Module):
    def __init__(self, config, hidden_dim=256, win_length=12, num_lstm_layer=3):
        super(LSTM_BN_no_concate, self).__init__()
        # here I'm going to assume the input is gonna be shaped as
        self.hidden_dim = hidden_dim
        self.win_length = win_length
        self.num_lstm_layer = num_lstm_layer
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(65, self.hidden_dim, num_lstm_layer, batch_first=True)
        self.output_mat1 = nn.Linear(self.hidden_dim, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.config = config
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_audio):
        mod_audio = input_audio
        # here I'm assuming that the input_audio is of proper shape
        hidden_state = [torch.zeros((self.num_lstm_layer, mod_audio.shape[0], self.hidden_dim)).to(self.config["device"]), 
        torch.zeros((self.num_lstm_layer, mod_audio.shape[0], self.hidden_dim)).to(self.config["device"])]
        out, hidden_state = self.lstm(mod_audio, hidden_state)
        # bn
        # x = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu(out)
        x = self.output_mat1(x)
        x = x[:, :, 0]
        x = self.sigmoid(x)
        return x
    def load_weights(self, pretrained_dict):
    #   not_copy = set(['fc.weight', 'fc.bias'])
        not_copy = set()
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in not_copy}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def test_forward(self, input_audio):
        x = self.forward(input_audio)
class LSTM_small(nn.Module):
    def __init__(self, config, input_dim=65, num_lstm_layer=1, bidirection=False, hidden_dim=256):
        super(LSTM_small, self).__init__()
        # here I'm going to assume the input is gonna be shaped as
        self.hidden_dim = hidden_dim
        self.num_lstm_layer = num_lstm_layer
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if bidirection:
            self.lstm = nn.LSTM(input_dim, self.hidden_dim, num_lstm_layer, batch_first=True, bidirectional=bidirection)
        else:
            self.lstm = nn.LSTM(input_dim, self.hidden_dim, num_lstm_layer, batch_first=True)
        if bidirection:
            self.output_mat1 = nn.Linear(self.hidden_dim * 2, 64)
        else:
            self.output_mat1 = nn.Linear(self.hidden_dim, 64)
        self.output_mat2 = nn.Linear(64, 1)
        self.bidirectional = bidirection
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.config = config
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_audio):
        mod_audio = input_audio
        # here I'm assuming that the input_audio is of proper shape
        if self.bidirectional:
            hidden_state = [torch.zeros((2*self.num_lstm_layer, mod_audio.shape[0], self.hidden_dim)).to(self.config["device"]), 
                        torch.zeros((2*self.num_lstm_layer, mod_audio.shape[0], self.hidden_dim)).to(self.config["device"])]
        else:
            hidden_state = [torch.zeros((self.num_lstm_layer, mod_audio.shape[0], self.hidden_dim)).to(self.config["device"]), 
                        torch.zeros((self.num_lstm_layer, mod_audio.shape[0], self.hidden_dim)).to(self.config["device"])]
        out, hidden_state = self.lstm(mod_audio, hidden_state)
        # bn
        # x = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu(out)
        x = self.output_mat1(x)
        x = self.relu(x)
        x = self.output_mat2(x)
        x = x[:, :, 0]
        x = self.sigmoid(x)
        return x
    def load_weights(self, pretrained_dict):
    #   not_copy = set(['fc.weight', 'fc.bias'])
        not_copy = set()
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in not_copy}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def test_forward(self, input_audio):
        x = self.forward(input_audio)

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

if __name__ == "__main__":
    config = export_config_Evan()
    load_model(config)