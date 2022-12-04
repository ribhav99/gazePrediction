from praatio import textgrid
import os
import parselmouth
import wandb
import utils
import yaml
import torch
import torch.nn as nn
import numpy as np
import wave
import torchaudio
from vanillaCNN import CNNet
import matplotlib.pyplot as plt
import random

def get_intervals(file_path):
    tg = textgrid.openTextgrid(file_path, False)
    return tg.tierDict['kijkrichting spreker1 [v] (TIE1)'].entryList

def compare_outputs(file_name):
    file_name = '../data/wav_files_5_seconds/_Number_0_channel_0_DVA3E.wav'
    # english = wave.open('../../../../Downloads/_Number_0_channel_0_DVA18AE_English_parsed.wav', 'rb')
    dutch = wave.open(file_name, 'rb')

    # length_english = english.getnframes() / english.getframerate()
    # frames_per_second_for_reading_english = english.getframerate() * english.getsampwidth()
    # frames_english = english.readframes(-1)
    # print(length_english, frames_per_second_for_reading_english)

    length_dutch = dutch.getnframes() / dutch.getframerate()
    frames_per_second_for_reading_dutch = dutch.getframerate() * dutch.getsampwidth()
    frames_dutch = dutch.readframes(-1)
    # print(length_dutch, frames_per_second_for_reading_dutch)

    # print(english.getparams())
    # print(dutch.getparams())
    ## TO PARSE AND SAVE MANUAL ENGLISH FILE
    # save_english = wave.open('../../../../Downloads/_Number_0_channel_0_DVA18AE_English_parsed.wav', 'wb')
    # save_english.setnchannels(1)
    # save_english.setsampwidth(2)
    # save_english.setframerate(48000)
    # save_english.writeframes(frames_english[:frames_per_second_for_reading_dutch * 5])

    # ##ENGLISH
    # waveform, sample_rate = torchaudio.load('../../../../Downloads/_Number_0_channel_0_DVA18AE_English_parsed.wav')
    # mfcc_spectogram_english = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)
    # # Intensity
    # snd = parselmouth.Sound('../../../../Downloads/_Number_0_channel_0_DVA18AE_English_parsed.wav')
    # intensity = torch.tensor(snd.to_intensity(time_step=0.01).values).flatten()
    # to_pad = mfcc_spectogram_english.shape[2] - intensity.shape[0]
    # intensity = torch.cat([intensity, torch.zeros(to_pad)], 0).to(torch.float32)
    # mfcc_spectogram_english = torch.cat([mfcc_spectogram_english, intensity.unsqueeze(0).unsqueeze(0)], 1)

    ##DUTCH
    waveform, sample_rate = torchaudio.load(file_name)
    mfcc_spectogram_dutch = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)
    # Intensity
    snd = parselmouth.Sound(file_name)
    intensity = torch.tensor(snd.to_intensity(time_step=0.01).values).flatten()
    to_pad = mfcc_spectogram_dutch.shape[2] - intensity.shape[0]
    intensity = torch.cat([intensity, torch.zeros(to_pad)], 0).to(torch.float32)
    mfcc_spectogram_dutch = torch.cat([mfcc_spectogram_dutch, intensity.unsqueeze(0).unsqueeze(0)], 1)

    # # print(mfcc_spectogram_english.shape, mfcc_spectogram_dutch.shape)

    #Load model
    run_obj = wandb.init(project="gaze_prediction", save_code=True,
                    resume='allow', id='288j7s6z')

    config_file_name = 'config.yaml'
    wandb.restore(config_file_name,
                            run_path=f'ribhav99/gaze_prediction/{run_obj.id}')
    config_file_path = utils.find_path(config_file_name, 'wandb')

    checkpoint_name = 'time=2022-11-29 06:22:13.018369_epoch=10.pt'
    wandb.restore(checkpoint_name,
                        run_path=f'ribhav99/gaze_prediction/{run_obj.id}')
    checkpoint_path = utils.find_path(checkpoint_name, 'wandb')

    '''
    This works but the loading config needs to be updated
    if new functions are used or added, eg: pool, act_fn, etc
    '''
    config = {}
    with open(config_file_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    for key in yaml_config:
        try:
            if 'value' in yaml_config[key]:
                config[key] = yaml_config[key]['value']
        except:
            pass

    config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
    # activation_fn
    if 'ReLU' in config["activation_fn"]:
        config["activation_fn"] = nn.ReLU()
    # pool
    start_ind = config['pool'].index('=') + 1
    end_ind = config['pool'].index(',')
    k_size = int(config['pool'][start_ind:end_ind])
    if 'AvgPool2d' in config['pool']:
        config['pool'] = nn.AvgPool2d(k_size)
    # loss_fn
    if 'weighted_binary_cross_entropy' in config['loss_fn']:
        config['loss_fn'] = utils.weighted_binary_cross_entropy
    '''
    Finished Loading config
    '''

    # Model
    model = CNNet(config, list(mfcc_spectogram_dutch.shape), int(5/config["window_length"]))
    pretrained_dict = torch.load(checkpoint_path, map_location=config['device'])
    model.load_weights(pretrained_dict)
    model.to(config['device'])
    model.eval()
    mfcc_spectogram_dutch = mfcc_spectogram_dutch.to(config['device'])
    # mfcc_spectogram_english = mfcc_spectogram_english.to(config['device'])
    dutch_pred = model(mfcc_spectogram_dutch)
    # english_pred = model(mfcc_spectogram_english)

    print('\n\n\n')
    print(dutch_pred)
    # print(english_pred)
    print('\n\n\n')

    dutch_pred = dutch_pred.cpu().detach().numpy().flatten()
    # english_pred = english_pred.cpu().detach().numpy().flatten()
    # print([torch.round(i) for i in dutch_pred])
    # print([torch.round(i) for i in english_pred])
    d = [np.round(i) for i in dutch_pred]
    # e = [torch.round(i) for i in english_pred]
    # e = [1 if i == 0.9 else 0 for i in e]
    plt.plot(range(50), d, label="dutch")
    plt.plot(range(50), dutch_pred, label="dutch_pred")
    # plt.plot(range(50), e, label="english")
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    '''
    1.
    '''
    # folder_path = '../data/gaze_files'
    # for file in os.listdir(folder_path):
    #     path = os.path.join(folder_path, file)
    #     try:
    #         get_intervals(path)
    #     except:
    #         print(file)

    '''
    2.
    '''
    # snd = parselmouth.Sound('../data/wav_files_5_seconds/_Number_0_channel_0_DVA1A.wav')
    # intensity = snd.to_intensity(time_step=0.01)
    # pitch = snd.to_pitch(time_step=0.01).to_array()
    # print(intensity.values.shape)
    # # print(pitch.squeeze(-1))
    # new = np.zeros(list(pitch.shape) + [2])
    # newer = []
    # for i in range(pitch.shape[0]):
    #     for j in range(pitch.shape[1]):
    #         new[i][j][0] = pitch[i][j][0] if not np.isnan(pitch[i][j][0]) else 0
    #         new[i][j][1] = pitch[i][j][1] if not np.isnan(pitch[i][j][1]) else 0

    # for i in range(new.shape[0]):
    #     newer.append(new[i, :, 0])
    #     newer.append(new[i, :, 1])
    # # void = np.void((np.float64(0), np.float64(0)))
    # # pitch_array = np.where(pitch[0] == np.float64('nan'), void, pitch)
    # newer = np.array(newer)
    # print(newer.shape)

    '''
    3.
    '''
    # run_obj = wandb.init(project="gaze_prediction", save_code=True,
    #             resume='allow', id='1j3g07a6')
    # checkpoint_name = 'config.yaml'
    # wandb.restore(checkpoint_name,
    #                         run_path=f'ribhav99/gaze_prediction/{run_obj.id}')
    # checkpoint_path = utils.find_path(checkpoint_name, 'wandb')
    # d = {}
    # with open(checkpoint_path, 'r') as f:
    #     yaml_config = yaml.safe_load(f)
    # for key in yaml_config:
    #     try:
    #         if 'value' in yaml_config[key]:
    #             d[key] = yaml_config[key]['value']
    #     except:
    #         pass
    
    # d["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
    # # activation_fn
    # if 'ReLU' in d["activation_fn"]:
    #     d["activation_fn"] = nn.ReLU()
    # # pool
    # start_ind = d['pool'].index('=') + 1
    # end_ind = d['pool'].index(',')
    # k_size = int(d['pool'][start_ind:end_ind])
    # if 'AvgPool2d' in d['pool']:
    #     d['pool'] = nn.AvgPool2d(k_size)
    # # loss_fn
    # if 'weighted_binary_cross_entropy' in d['loss_fn']:
    #     d['loss_fn'] = utils.weighted_binary_cross_entropy

    # print(type(d['pool']))
    # print(d['pool'])
    # print(type(d['load_model']))
    # print(d['load_model'])


    # '''
    # 4
    # '''
    # torch.set_printoptions(profile="full")
    # snd = parselmouth.Sound('../data/wav_files_single_channel/channel_0_DVA1A.wav')
    # intensity = torch.tensor(snd.to_intensity(time_step=0.1).values).flatten()
    # print(torch.median(intensity))

    '''
    5
    '''
    # files = os.listdir('../data/wav_files_5_seconds')
    # random.shuffle(files)
    # for f in files:
    #     print(f)
    #     compare_outputs(f)
    compare_outputs('../data/wav_files_5_seconds/_Number_35_channel_1_DVB12T.wav')