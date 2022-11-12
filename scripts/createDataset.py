import torchaudio
import os
from tqdm import tqdm
import torch
import warnings
import utils
from readGazeFiles import create_targets_for_all_participants
warnings.filterwarnings("ignore") 

def load_audio_data(wav_dir):
    all_mfcc = {}
    for file_name in tqdm(os.listdir(wav_dir), key=utils.sort_name_by_part_number):
        participant = utils.get_participant_id_from_audio_clips(file_name)
        waveform, sample_rate = torchaudio.load(os.path.join(wav_dir, file_name))
        mfcc_spectogram = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)
        if participant not in all_mfcc:
            all_mfcc[participant] = mfcc_spectogram
        else:
            all_mfcc[participant] = torch.cat([all_mfcc[participant], mfcc_spectogram], dim=0)
    
    return all_mfcc

class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, all_mfcc, all_targets, audio_length=5, window_length=0.1):
        for key in all_mfcc:
            if key not in all_targets:
                del all_mfcc[key]

        self.all_mfcc = all_mfcc
        self.audio_length = audio_length
        self.window_length = window_length
        self.all_targets = all_targets

        num_x= sum([self.all_mfcc[i].shape[0] for i in self.all_mfcc])
        num_y = sum([self.all_targets[i].shape[0] for i in self.all_targets])
        assert num_x == num_y

        self.size = num_x
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    wav_5_sec_dir = '../data/wav_files_5_seconds/'
    gaze_dir = '../data/gaze_files'
    print('Loading Audio Files and Computing MFCC Spectogram')
    all_mfcc = load_audio_data(wav_5_sec_dir)
    all_targets = create_targets_for_all_participants(gaze_dir, 5, 0.1)
    # Initialise Dataset
    del all_mfcc
    del all_targets