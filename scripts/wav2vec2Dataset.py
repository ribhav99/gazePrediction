import torchaudio
import os
from tqdm import tqdm
import torch
import warnings
import utils
from readGazeFiles import create_targets_for_all_participants
warnings.filterwarnings("ignore") 

def load_audio_data(wav_dir, participants=None, time_step=0.1):
    all_waveforms = {}
    for file_name in tqdm(sorted(os.listdir(wav_dir), key=utils.sort_name_by_part_number)):
        participant, channel = utils.get_participant_id_from_audio_clips(file_name)
        full_key = participant + '_' + channel
        if participants is not None:
            if participant not in participants:
                continue

        waveform, sample_rate = torchaudio.load(os.path.join(wav_dir, file_name))
        
        if full_key not in all_waveforms:
            all_waveforms[full_key] = waveform
        else:
            all_waveforms[full_key] = torch.cat([all_waveforms[full_key], waveform], dim=0)
    
    return all_waveforms


class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, wav_dir, gaze_dir, audio_length=5, window_length=0.1, time_step=0.1):
        super().__init__()
        print('Initialising Targets')
        all_targets = create_targets_for_all_participants(gaze_dir, audio_length, window_length) 
        participants = [i[:i.index('.gaze')] for i in os.listdir(gaze_dir)]
        print('Initialising Data')
        waveform = load_audio_data(wav_dir, participants, time_step)
        to_delete = []
        for key in waveform:
            if key not in all_targets:
                print(f'Deleting: {key}. Should never reach here though')
                to_delete.append(key)
        for i in to_delete:
            del waveform[i]
        print('Finished Initialising Targets')
        print(waveform[0].shape)
        num_x = [waveform[i].shape[0] for i in waveform]
        num_y = [all_targets[i].shape[0] for i in all_targets]
        assert sum(num_x) == sum(num_y)

        self.audio_length = audio_length
        self.window_length = window_length
        self.size = sum(num_x)

        print('Matching data with targets')
        self.concated_waveforms = None
        self.concated_targets = None
        for key in waveform:
            if self.concated_waveforms is None:
                self.concated_waveforms = waveform[key]
            else:
                self.concated_waveforms = torch.cat([self.concated_waveforms, waveform[key]], dim=0)
            
            if self.concated_targets is None:
                self.concated_targets = all_targets[key]
            else:
                self.concated_targets = torch.cat([self.concated_targets, all_targets[key]], dim=0)

        assert self.concated_waveforms.shape[0] == self.concated_targets.shape[0]


    def __len__(self):
        return self.size
    

    def __getitem__(self, idx):
        return utils.normalise_tensor(self.concated_waveforms[idx]), self.concated_targets[idx]


if __name__ == '__main__':

    wav_5_sec_dir = '../data/wav_files_5_seconds/'
    gaze_dir = '../data/gaze_files'
    print('Initialising Dataset')
    dataset = AudioDataset(wav_5_sec_dir, gaze_dir, 5, 0.1, 0.01)

    print(dataset.__len__())
    x, y = dataset.__getitem__(420)
    print(x.shape)
    print(y.shape)
    print(x.dtype)
    print(y.dtype)
    