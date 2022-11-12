import torchaudio
import os
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings("ignore") 

def get_participant_id(file_name):
    start_index = file_name.rfind('_') + 1
    end_index = file_name.index('.wav')
    return file_name[start_index:end_index]

def load_audio_data(wav_dir):
    all_mfcc = {}
    for file_name in tqdm(os.listdir(wav_dir)):
        participant = get_participant_id(file_name)
        waveform, sample_rate = torchaudio.load(os.path.join(wav_dir, file_name))
        mfcc_spectogram = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)
        if participant not in all_mfcc:
            all_mfcc[participant] = mfcc_spectogram
        else:
            all_mfcc[participant] = torch.cat([all_mfcc[participant], mfcc_spectogram], dim=0)
    
    return all_mfcc

if __name__ == '__main__':
    wav_dir = '../data/wav_files_5_seconds/'
    # all_mfcc = load_audio_data(wav_dir)
    for i in sorted(os.listdir(wav_dir)):
        print(i)