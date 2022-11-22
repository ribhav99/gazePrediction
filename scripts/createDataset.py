import torchaudio
import os
from tqdm import tqdm
import torch
import warnings
import utils
from readGazeFiles import create_targets_for_all_participants
import parselmouth
warnings.filterwarnings("ignore") 

def load_audio_data(wav_dir, participants=None, time_step=0.1):
    all_mfcc = {}
    for file_name in tqdm(sorted(os.listdir(wav_dir), key=utils.sort_name_by_part_number)):
        participant, channel = utils.get_participant_id_from_audio_clips(file_name)
        full_key = participant + '_' + channel
        if participants is not None:
            if participant not in participants:
                continue

        waveform, sample_rate = torchaudio.load(os.path.join(wav_dir, file_name))
        mfcc_spectogram = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)
        snd = parselmouth.Sound(os.path.join(wav_dir, file_name))
        intensity = torch.tensor(snd.to_intensity(time_step=time_step).values).flatten()
        to_pad = mfcc_spectogram.shape[2] - intensity.shape[0]
        intensity = torch.cat([intensity, torch.zeros(to_pad)], 0).to(torch.float32)
        mfcc_spectogram = torch.cat([mfcc_spectogram, intensity.unsqueeze(0).unsqueeze(0)], 1)
        if full_key not in all_mfcc:
            all_mfcc[full_key] = mfcc_spectogram
        else:
            all_mfcc[full_key] = torch.cat([all_mfcc[full_key], mfcc_spectogram], dim=0)
    
    return all_mfcc


class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, wav_dir, gaze_dir, audio_length=5, window_length=0.1, time_step=0.1):
        super().__init__()
        print('Initialising Targets')
        all_targets = create_targets_for_all_participants(gaze_dir, audio_length, window_length) 
        participants = [i[:i.index('.gaze')] for i in os.listdir(gaze_dir)]
        print('Initialising Data')
        # targets_shape = int(audio_length / window_length)
        all_mfcc = load_audio_data(wav_dir, participants)
        to_delete = []
        for key in all_mfcc:
            if key not in all_targets:
                print(f'Deleting: {key}. Should never reach here though')
                to_delete.append(key)
        for i in to_delete:
            del all_mfcc[i]
        
        num_x = [all_mfcc[i].shape[0] for i in all_mfcc]
        num_y = [all_targets[i].shape[0] for i in all_targets]
        assert sum(num_x) == sum(num_y)

        self.audio_length = audio_length
        self.window_length = window_length
        self.size = sum(num_x)

        self.concated_mfcc = None
        self.concated_targets = None
        for key in all_mfcc:
            if self.concated_mfcc is None:
                self.concated_mfcc = all_mfcc[key]
            else:
                self.concated_mfcc = torch.cat([self.concated_mfcc, all_mfcc[key]], dim=0)
            
            if self.concated_targets is None:
                self.concated_targets = all_targets[key]
            else:
                self.concated_targets = torch.cat([self.concated_targets, all_targets[key]], dim=0)

        assert self.concated_mfcc.shape[0] == self.concated_targets.shape[0]


    def __len__(self):
        return self.size
    

    def __getitem__(self, idx):
        return utils.normalise_tensor(self.concated_mfcc[idx]), self.concated_targets[idx]


if __name__ == '__main__':

    wav_5_sec_dir = '../data/wav_files_5_seconds/'
    gaze_dir = '../data/gaze_files'
    print('Initialising Dataset')
    dataset = AudioDataset(wav_5_sec_dir, gaze_dir, 5, 0.1, 0.1)

    print(dataset.__len__())
    x, y = dataset.__getitem__(420)
    print(x.shape)
    print(y.shape)
    print(x.dtype)
    print(y.dtype)
    