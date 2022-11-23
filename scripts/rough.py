from praatio import textgrid
import os
import parselmouth
import wandb
import utils
import yaml
import torch
import torch.nn as nn

def get_intervals(file_path):
    tg = textgrid.openTextgrid(file_path, False)
    return tg.tierDict['kijkrichting spreker1 [v] (TIE1)'].entryList


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
    # intensity = snd.to_intensity(time_step=0.1)
    # pitch = snd.to_pitch(time_step=0.1)
    # print(pitch.to_array())

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

    
