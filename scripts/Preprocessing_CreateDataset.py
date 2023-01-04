import torchaudio
import os
from tqdm import tqdm
import torch
import warnings
import wave
import utils
from readGazeFiles import create_targets_for_all_participants
from readAudioFiles import convert_to_mono_channel, create_audio_data
import parselmouth
import numpy as np
import shutil
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
        # Intensity
        snd = parselmouth.Sound(os.path.join(wav_dir, file_name))
        intensity = torch.tensor(snd.to_intensity(time_step=time_step).values).flatten()
        to_pad = mfcc_spectogram.shape[2] - intensity.shape[0]
        intensity = torch.cat([intensity, torch.zeros(to_pad)], 0).to(torch.float32)
        mfcc_spectogram = torch.cat([mfcc_spectogram, intensity.unsqueeze(0).unsqueeze(0)], 1)
        # Pitch
        # pitch = snd.to_pitch(time_step=time_step).to_array()
        # new = np.zeros(list(pitch.shape) + [2])
        # newer = []
        # for i in range(pitch.shape[0]):
        #     for j in range(pitch.shape[1]):
        #         new[i][j][0] = pitch[i][j][0] if not np.isnan(pitch[i][j][0]) else 0
        #         new[i][j][1] = pitch[i][j][1] if not np.isnan(pitch[i][j][1]) else 0
        # for i in range(new.shape[0]):
        #     newer.append(new[i, :, 0])
        #     newer.append(new[i, :, 1])
        # newer = torch.tensor(newer)
        # to_pad = mfcc_spectogram.shape[2] - newer.shape[1]
        # newer = torch.cat([newer, torch.zeros((newer.shape[0], to_pad))], 1).to(torch.float32)
        # mfcc_spectogram = torch.cat([mfcc_spectogram, newer.unsqueeze(0)], 1)
        if full_key not in all_mfcc:
            all_mfcc[full_key] = mfcc_spectogram
        else:
            all_mfcc[full_key] = torch.cat([all_mfcc[full_key], mfcc_spectogram], dim=0)
    return all_mfcc

def preprocess_data(wav_dir, gaze_dir, audio_length, window_length=0.1, time_step=0.1, testing=False):
    # get target data
    print('Initialising Targets')
    shutil.rmtree("../data/processed_file/target")
    shutil.rmtree("../data/processed_file/input")
    os.mkdir("../data/processed_file/target")
    os.mkdir("../data/processed_file/input")
    all_targets = create_targets_for_all_participants(gaze_dir, audio_length, window_length) 
    participants = [i[:i.index('.gaze')] for i in os.listdir(gaze_dir)]
    keys = list(all_targets.keys())
    file_names = []
    # iterate through the targets and save each one as a file
    for key in keys:
        for i in range(0, int(all_targets[key].shape[0])):
            file_name = "../data/processed_file/target/{}_{}.pt"
            file_name = file_name.format(key, i)
            torch.save(all_targets[key][i], file_name)
            file_names.append(file_name)
    print(file_names.__len__())
    # get sound_data
    # first convert to single channel wav files
    if not testing: # ignore this when testing
        convert_to_mono_channel(wav_dir, '../data/wav_files', 0)
        convert_to_mono_channel(wav_dir, '../data/wav_files', 1)
    # then parse then into 5 seconds segments
    for key in keys:
        # just the part with the DVA13U
        file_raw_name = key.split("_")[0]
        # do it for both channels:
        for cha in range(0, 2):
            input_file_name = "../data/wav_files_single_channel/channel_{}_{}.wav".format(cha, file_raw_name)
            output_file_name_per_clip = "../data/processed_file/input/{}_channel_{}_".format(file_raw_name, cha)
            output_file_name_per_clip = output_file_name_per_clip + "{}.pt"
            wav = wave.open(input_file_name, 'rb')
            length = wav.getnframes() / wav.getframerate()
            frames_per_second_for_reading = wav.getframerate() * wav.getsampwidth()
            frames = wav.readframes(-1)
            for i in range(int(length/audio_length)):
                save_path = output_file_name_per_clip.format(i)
                outwav = wave.open(save_path, 'wb')
                outwav.setparams(wav.getparams())
                outwav.setnframes(frames_per_second_for_reading * audio_length)
                outwav.writeframes(frames[frames_per_second_for_reading * audio_length*i:frames_per_second_for_reading * audio_length*(i+1)])
                outwav.close()
    print(len(os.listdir("../data/processed_file/input/")))
            

        
            
                    
    

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, wav_dir, gaze_dir, audio_length=5, window_length=0.1, time_step=0.1):
        super().__init__()
        print('Initialising Targets')
        all_targets = create_targets_for_all_participants(gaze_dir, audio_length, window_length) 
        participants = [i[:i.index('.gaze')] for i in os.listdir(gaze_dir)]
        print('Initialising Data')
        # targets_shape = int(audio_length / window_length)
        all_mfcc = load_audio_data(wav_dir, participants, time_step)
        to_delete = []
        for key in all_mfcc:
            if key not in all_targets:
                print(f'Deleting: {key}. Should never reach here though')
                to_delete.append(key)
        for i in to_delete:
            del all_mfcc[i]
        num_x = [all_mfcc[i].shape[0] for i in all_mfcc]
        num_y = [all_targets[i].shape[0] for i in all_targets]
        print(sum(num_x), sum(num_y))
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

    # metadata: length of each audio segments
    length_of_training_audio_clips = 5

    # process the wav files
    # if len(os.listdir('../data/wav_files_single_channel/')) == 0:
    wav_files = os.listdir("../data/wav_files_5_seconds")
    gaze_files = os.listdir("../data/gaze_files")
    # gazes = create_targets_for_all_participants('../data/wav_files_5_seconds', length_of_training_audio_clips, 0.1)
    # print(len(gazes.keys()))
    
    wav_5_sec_dir = '../data/wav_files_5_seconds/'
    gaze_dir = '../data/gaze_files'
    
    preprocess_data(wav_5_sec_dir, gaze_dir, length_of_training_audio_clips, 0.1, 0.01, testing=True)
    
    A[2]
    wav_folder = '../data/wav_files_single_channel/'
    convert_to_mono_channel(wav_folder, '../data/wav_files', 0)
    convert_to_mono_channel(wav_folder, '../data/wav_files', 1)
    for file in tqdm(os.listdir(wav_folder)):
        path = os.path.join(wav_folder, file)
        create_audio_data(path, '../data/wav_files_5_seconds', length_of_training_audio_clips)
    print(len(os.listdir("../data/wav_files_5_seconds")))
    
    A[2]
    dataset = AudioDataset(wav_5_sec_dir, gaze_dir, length_of_training_audio_clips, 0.1, 0.01)

    print(dataset.__len__())
    x, y = dataset.__getitem__(420)
    print(x.shape)
    print(y.shape)
    print(x.dtype)
    print(y.dtype)
    