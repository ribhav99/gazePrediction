from praatio import textgrid
import os
import parselmouth
import wandb
import utils
import yaml

def get_intervals(file_path):
    tg = textgrid.openTextgrid(file_path, False)
    return tg.tierDict['kijkrichting spreker1 [v] (TIE1)'].entryList


if __name__ == '__main__':
    # folder_path = '../data/gaze_files'
    # for file in os.listdir(folder_path):
    #     path = os.path.join(folder_path, file)
    #     try:
    #         get_intervals(path)
    #     except:
    #         print(file)


    # snd = parselmouth.Sound('../data/wav_files_5_seconds/_Number_0_channel_0_DVA1A.wav')
    # intensity = snd.to_intensity(time_step=0.1)
    # pitch = snd.to_pitch(time_step=0.1)
    # print(pitch.to_array())

    run_obj = wandb.init(project="gaze_prediction", save_code=True,
                resume='allow', id='1j3g07a6')
    checkpoint_name = 'config.yaml'
    wandb.restore(checkpoint_name,
                            run_path=f'ribhav99/gaze_prediction/{run_obj.id}')
    checkpoint_path = utils.find_path(checkpoint_name, 'wandb')
    d = {}
    with open(checkpoint_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    for key in yaml_config:
        try:
            if 'value' in yaml_config[key]:
                d[key] = yaml_config[key]['value']
        except:
            pass
    
    print(type(d['pool']))
    print(d['pool'])

    # import torch
    # print(torch.cuda.is_available())
    # x = []
    # y = torch.ones(5).tolist()
    # print(x + y)