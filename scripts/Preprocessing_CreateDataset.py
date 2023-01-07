import torchaudio
import os
from tqdm import tqdm
import torch
import warnings
import wave
import utils
import librosa
import textgrids
import math
from bisect import bisect
from readGazeFiles import create_targets_for_all_participants
from readAudioFiles import convert_to_mono_channel, create_audio_data
import parselmouth
import python_speech_features as psf
import numpy as np
import shutil
from config.config import export_config_Evan
warnings.filterwarnings("ignore") 
def pre_process_audio_data(wav_dir, participants=None, time_step=0.1):
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
def prepare_data(wav_dir, gaze_dir, length_of_training_audio_clips, sample_width=0.1, time_step=0.1, testing=False):
    # get target data
    print('Initialising Targets')
    try:
        shutil.rmtree("../data/processed_file")
    except:
        pass
    os.mkdir("../data/processed_file")
    os.mkdir("../data/processed_file/target")
    os.mkdir("../data/processed_file/input")

    # iterate through each audio/annotation pair
    audio_file_list = list(os.listdir("../data/wav_files"))
    annotation_file_list = list(os.listdir("../data/gaze_files"))

    training_sample_list = []
    for i in range(0, len(annotation_file_list)):
        annotation_filepath = annotation_file_list[i]
        annotation_speaker_name = annotation_filepath.split(".")[0].split("_")[0]
        for j in range(0, len(audio_file_list)):
            audio_filepath = audio_file_list[j]
            audio_speaker_name = audio_filepath.split(".")[0]
            # find the file that matches the annotation
            if audio_speaker_name == annotation_speaker_name:
                # load praat_script
                try:
                    grid = textgrids.TextGrid(os.path.join(gaze_dir, annotation_filepath))
                except:
                    # failutre to load praat script
                    print("failure to load {}".format(annotation_filepath))
                    break
                # load audio
                audio, sr = librosa.load(os.path.join(wav_dir, audio_filepath), mono=False)
                try:
                    speaker_0_gaze_tier = grid['kijkrichting spreker1 [v] (TIE1)']
                    speaker_1_gaze_tier = grid['kijkrichting spreker2 [v] (TIE3)']
                except:
                    speaker_0_gaze_tier = grid['kijkrichting spreker1 (TIE1)']
                    speaker_1_gaze_tier = grid['kijkrichting spreker2 (TIE3)']

                # get the list of lower bound of the intervals
                annotation_speaker_0_starts = [i.xmin for i in speaker_0_gaze_tier]
                annotation_speaker_1_starts = [i.xmin for i in speaker_1_gaze_tier]
                # obtain total number of segments:
                end_time = speaker_1_gaze_tier[-1].xmax
                num_of_segments = math.floor(end_time / length_of_training_audio_clips)
                num_targets_per_segment = int(length_of_training_audio_clips / sample_width)
                num_samples_per_segment = int(sr * length_of_training_audio_clips)
                for s in range(num_of_segments):
                    # get speaker_target
                    target_speaker_0 = torch.zeros([num_targets_per_segment])
                    target_speaker_1 = torch.zeros([num_targets_per_segment])
                    for sample in range(num_targets_per_segment):
                        time = (s * length_of_training_audio_clips) + (sample * sample_width)

                        index_speaker0 = bisect(annotation_speaker_0_starts, time)
                        target0 = 1 if speaker_0_gaze_tier[index_speaker0 - 1].text == 'g' else 0 # g is gaze, x is aversion
                        target_speaker_0[sample] = target0

                        index_speaker1 = bisect(annotation_speaker_1_starts, time)
                        target1 = 1 if speaker_1_gaze_tier[index_speaker1 - 1].text == 'g' else 0 # g is gaze, x is aversion
                        target_speaker_1[sample] = target1
                    # get the wav files
                    wav_speaker_0 = audio[0, s * num_samples_per_segment: (s + 1) * num_samples_per_segment]
                    wav_speaker_1 = audio[1, s * num_samples_per_segment: (s + 1) * num_samples_per_segment]
                    # pad the speaker_array with zero at the front abd back
                    wav_speaker_0 = np.pad(wav_speaker_0, int(sr*time_step/2))
                    wav_speaker_1 = np.pad(wav_speaker_1, int(sr*time_step/2))
                    # get the mfcc features for speaker 0
                    winstep = int(math.floor(time_step * sr))
                    mfcc_featspeaker_0 = psf.mfcc(wav_speaker_0, samplerate=sr, winlen=config["window_length"],
                                         winstep=config["window_length"], nfft=winstep, numcep=13)
                    logfbank_featspeaker_0 = psf.logfbank(wav_speaker_0, samplerate=sr, winlen=config["window_length"],
                                                 winstep=config["window_length"],nfft=winstep, nfilt=26)
                    ssc_featspeaker_0 = psf.ssc(wav_speaker_0, samplerate=sr, winlen=config["window_length"],
                                       winstep=config["window_length"], nfft=winstep, nfilt=26)
                    full_feat_speaker_0 = np.concatenate([mfcc_featspeaker_0, logfbank_featspeaker_0, ssc_featspeaker_0], axis=1)
                    # get the mfcc features for speaker 1
                    mfcc_featspeaker_1 = psf.mfcc(wav_speaker_1, samplerate=sr, winlen=config["window_length"],
                                                  winstep=config["window_length"], nfft=winstep, numcep=13)
                    logfbank_featspeaker_1 = psf.logfbank(wav_speaker_1, samplerate=sr, winlen=config["window_length"],
                                                          winstep=config["window_length"], nfft=winstep, nfilt=26)
                    ssc_featspeaker_1 = psf.ssc(wav_speaker_1, samplerate=sr, winlen=config["window_length"],
                                                winstep=config["window_length"], nfft=winstep, nfilt=26)
                    full_feat_speaker_1 = np.concatenate(
                        [mfcc_featspeaker_1, logfbank_featspeaker_1, ssc_featspeaker_1], axis=1)
                    print(full_feat_speaker_1.shape)
                    # store the files
                    torch.save(full_feat_speaker_0, "../data/processed_file/input/{}_part_{}_speaker_0.pt".format(annotation_speaker_name, s))
                    torch.save(full_feat_speaker_1, "../data/processed_file/input/{}_part_{}_speaker_1.pt".format(annotation_speaker_name, s))
                    torch.save(target_speaker_0, "../data/processed_file/target/{}_part_{}_speaker_0.pt".format(annotation_speaker_name, s))
                    torch.save(target_speaker_1, "../data/processed_file/target/{}_part_{}_speaker_1.pt".format(annotation_speaker_name, s))
                    training_sample_list.append("{}_part_{}".format(annotation_speaker_name, s))
                with open("../data/processed_file/index.txt", "w") as f:
                    for line in range(0, len(training_sample_list)):
                        f.write(training_sample_list[line])
                        f.write("\n")
                    f.close()
                break

class AudioDataset_Evan(torch.utils.data.Dataset):
    def __init__(self, data_dir, audio_length, window_length=0.01, time_step=0.01, listener_data=False):
        super().__init__()
        print('Initialising Targets')
        data_indexes = []
        with open(os.path.join(data_dir, "index.txt")) as f:
            data_indexes = f.readlines()
            f.close()
        for i in range(0, len(data_indexes)):
            data_indexes[i] = data_indexes[i].strip("\n")
        self.data_paths = data_indexes
        self.data_dir = data_dir
        self.listener_data = listener_data
    def __len__(self):
        return len(self.data_paths) * 2
    def __getitem__(self, idx):
        channel = 0
        if idx >= len(self.data_paths):
            idx = idx // 2
            channel = 1
        file_name = self.data_paths[idx]
        input_file_path = os.path.join(self.data_dir, "input/{}_speaker_{}.pt".format(file_name, channel))
        target_file_path = os.path.join(self.data_dir, "target/{}_speaker_{}.pt".format(file_name, channel))
        if not self.listener_data:
            input = torch.load(input_file_path)
            target = torch.load(target_file_path)
            return input, target

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, audio_length=5, window_length=0.1, time_step=0.1):
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

    wav_5_sec_dir = '../data/wav_files_5_seconds/'
    gaze_dir = '../data/gaze_files'
    wav_dir = '../data/wav_files'
    data_dir = "../data/processed_file"
    cold_start = True

    config = export_config_Evan()

    if cold_start:
        prepare_data(wav_dir, gaze_dir,
                     config["sample_length"],
                     config["window_length"],
                     config["time_step"], testing=False)
    else:
        pass
    dataset = AudioDataset_Evan(data_dir, 5, 0.01, 0.01)
    data, target = dataset.__getitem__(0)