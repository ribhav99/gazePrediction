import torchaudio
import os
from tqdm import tqdm
import torch
import warnings
import wave
import utils
import librosa
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
def prepare_data_wrong(wav_dir, gaze_dir, length_of_training_audio_clips, sample_width=0.1, time_step=0.1, testing=False):
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
                try:
                    speaker_0_gaze_tier = grid['kijkrichting spreker1 [v] (TIE1)']
                    speaker_1_gaze_tier = grid['kijkrichting spreker2 [v] (TIE3)']
                    speaker_0_transcript_tier = grid['spreker1 [v] (TIE0)']
                    speaker_1_transcript_tier = grid['spreker2 [v] (TIE2)']
                except:
                    speaker_0_gaze_tier = grid['kijkrichting spreker1 (TIE1)']
                    speaker_1_gaze_tier = grid['kijkrichting spreker2 (TIE3)']
                    speaker_0_transcript_tier = grid['spreker1 [v] (TIE0)']
                    speaker_1_transcript_tier = grid['spreker2 [v] (TIE2)']
                # load audio
                audio, sr = librosa.load(os.path.join(wav_dir, audio_filepath), mono=False, sr=config["sample_rate"])
                # get the list of lower bound of the intervals
                annotation_speaker_0_starts = [i.xmin for i in speaker_0_gaze_tier]
                annotation_speaker_1_starts = [i.xmin for i in speaker_1_gaze_tier]
                transcript_speaker_0_starts = [i.xmin for i in speaker_0_transcript_tier]
                transcript_speaker_1_starts = [i.xmin for i in speaker_1_transcript_tier]
                
                # obtain total number of segments:
                end_time = speaker_1_gaze_tier[-1].xmax
                num_of_segments = math.floor(end_time / length_of_training_audio_clips)
                num_targets_per_segment = int(length_of_training_audio_clips / sample_width)
                num_samples_per_segment = int(sr * length_of_training_audio_clips)
                for s in range(num_of_segments):
                    # get speaker_target
                    target_speaker_0 = torch.zeros([num_targets_per_segment])
                    target_speaker_1 = torch.zeros([num_targets_per_segment])
                    #   speaking = 1, silent = 0
                    #   absolute time of current turn
                    #   normalized time since turn started
                    #   normalized time till turn end
                    #   time till next speech start
                    #   time till next pause start
                    #   time since last speech end
                    #   time since last pause end
                    turn_taking_feature_vector0 = torch.zeros([num_targets_per_segment, 8])
                    turn_taking_feature_vector1 = torch.zeros([num_targets_per_segment, 8])
                    for sample in range(num_targets_per_segment):
                        time = (s * length_of_training_audio_clips) + (sample * sample_width)

                        index_speaker0 = bisect(annotation_speaker_0_starts, time)
                        target0 = 1 if speaker_0_gaze_tier[index_speaker0 - 1].text == 'g' else 0 # g is gaze, x is aversion
                        target_speaker_0[sample] = target0

                        index_speaker1 = bisect(annotation_speaker_1_starts, time)
                        target1 = 1 if speaker_1_gaze_tier[index_speaker1 - 1].text == 'g' else 0 # g is gaze, x is aversion
                        target_speaker_1[sample] = target1

                        index_transcript_speaker0 = bisect(transcript_speaker_0_starts, time) - 1
                        # record whether the person is speaking (speaker 0)
                        if index_transcript_speaker0 >= 0:
                            if speaker_0_transcript_tier[index_transcript_speaker0].text == "":
                                speaking_0 = 0 # feature 0
                            else:
                                speaking_0 = 1 # feature 0
                            start = speaker_0_transcript_tier[index_transcript_speaker0].xmin
                            end = speaker_0_transcript_tier[index_transcript_speaker0].xmax
                        else:
                            index_transcript_speaker0 = 0
                            speaking_0 = 0
                            start = 0
                            end = speaker_0_transcript_tier[0].xmin
                        # record property of the current interval
                        interval_length_0 = end - start # feature 1
                        time_since_start_0 = time - start # feature 2
                        time_till_end_0 = end - time # feature 3
                        time_till_next_speech_0 = -1 # feature 4
                        for interval in range(index_transcript_speaker0+1, len(speaker_0_transcript_tier)):
                            if speaker_0_transcript_tier[interval].text != "":
                                time_till_next_speech_0 = speaker_0_transcript_tier[interval].xmin - time
                                break
                        if time_till_next_speech_0 < 0:
                            time_till_next_speech_0 = (end + 1) - time 
                        time_till_next_slience_0 = -1 # feature 5
                        for interval in range(index_transcript_speaker0+1, len(speaker_0_transcript_tier)):
                            if speaker_0_transcript_tier[interval].text == "":
                                time_till_next_slience_0 = speaker_0_transcript_tier[interval].xmin - time
                                break
                        if time_till_next_slience_0 < 0:
                            time_till_next_slience_0 = (end + 1) - time 
                        time_till_prev_speech_0 = -1 # feature 6
                        for interval in range(index_transcript_speaker0-1, -1, -1):
                            if speaker_0_transcript_tier[interval].text != "":
                                time_till_prev_speech_0 = time - speaker_0_transcript_tier[interval].xmax
                                break
                        if time_till_prev_speech_0 < 0:
                            time_till_prev_speech_0 = time - (start - 1)
                        time_till_prev_silence_0 = -1 # feature 7
                        for interval in range(index_transcript_speaker0-1, -1, -1):
                            if speaker_0_transcript_tier[interval].text == "":
                                time_till_prev_silence_0 = time - speaker_0_transcript_tier[interval].xmax
                                break
                        if time_till_prev_silence_0 < 0:
                            time_till_prev_silence_0 = time - (start - 1)
                        turn_taking_feature_vector0[sample, 0] = speaking_0
                        turn_taking_feature_vector0[sample, 1] = interval_length_0
                        turn_taking_feature_vector0[sample, 2] = time_since_start_0 
                        turn_taking_feature_vector0[sample, 3] = time_till_end_0
                        turn_taking_feature_vector0[sample, 4] = time_till_next_speech_0
                        turn_taking_feature_vector0[sample, 5] = time_till_next_slience_0
                        turn_taking_feature_vector0[sample, 6] = time_till_prev_speech_0
                        turn_taking_feature_vector0[sample, 7] = time_till_prev_silence_0
                        index_transcript_speaker1 = bisect(transcript_speaker_1_starts, time) - 1
                        if index_transcript_speaker1 >= 0:
                            # record whether the person is speaking (speaker 1)
                            if speaker_1_transcript_tier[index_transcript_speaker1].text == "":
                                speaking_1 = 0 # feature 0
                            else:
                                speaking_1 = 1 # feature 0
                            start = speaker_1_transcript_tier[index_transcript_speaker1].xmin
                            end = speaker_1_transcript_tier[index_transcript_speaker1].xmax
                        
                        else:
                            index_transcript_speaker1 = 0
                            speaking_1 = 0
                            start = 0
                            end = speaker_1_transcript_tier[0].xmin
                        # record property of the current interval
                        interval_length_1 = end - start # feature 1
                        time_since_start_1 = time - start # feature 2
                        time_till_end_1 = end - time # feature 3
                        time_till_next_speech_1 = -1 # feature 4
                        for interval in range(index_transcript_speaker1+1, len(speaker_1_transcript_tier)):
                            if speaker_1_transcript_tier[interval].text != "":
                                time_till_next_speech_1 = speaker_1_transcript_tier[interval].xmin - time
                                break
                        if time_till_next_speech_1 < 0:
                            time_till_next_speech_1 = end - time + 1
                        time_till_next_slience_1 = -1 # feature 5
                        # get the first interval with silence before the current interval
                        for interval in range(index_transcript_speaker1+1, len(speaker_1_transcript_tier)):
                            if speaker_1_transcript_tier[interval].text == "":
                                time_till_next_slience_1 = speaker_1_transcript_tier[interval].xmin - time
                                break
                        if time_till_next_slience_1 == -1:
                            time_till_next_slience_1 = end - time + 1
                        time_till_prev_speech_1 = -1 # feature 6
                        for interval in range(index_transcript_speaker1-1, -1, -1):
                            if speaker_1_transcript_tier[interval].text != "":
                                time_till_prev_speech_1 = time - speaker_1_transcript_tier[interval].xmax 
                                break
                        if time_till_prev_speech_1 < 0:
                            time_till_prev_speech_1 = time - start + 1
                        time_till_prev_silence_1 = -1 # feature 7
                        for interval in range(index_transcript_speaker1-1, -1, -1):
                            if speaker_1_transcript_tier[interval].text == "":
                                time_till_prev_silence_1 = time - speaker_1_transcript_tier[interval].xmax 
                                break
                        if time_till_prev_silence_1 < 0:
                            time_till_prev_silence_1 = time - start + 1
                        turn_taking_feature_vector1[sample, 0] = speaking_1
                        turn_taking_feature_vector1[sample, 1] = interval_length_1
                        turn_taking_feature_vector1[sample, 2] = time_since_start_1
                        turn_taking_feature_vector1[sample, 3] = time_till_end_1
                        turn_taking_feature_vector1[sample, 4] = time_till_next_speech_1
                        turn_taking_feature_vector1[sample, 5] = time_till_next_slience_1
                        turn_taking_feature_vector1[sample, 6] = time_till_prev_speech_1
                        turn_taking_feature_vector1[sample, 7] = time_till_prev_silence_1
                    # get the wav files
                    wav_speaker_0 = audio[0, s * num_samples_per_segment: (s + 1) * num_samples_per_segment]
                    wav_speaker_1 = audio[1, s * num_samples_per_segment: (s + 1) * num_samples_per_segment]
                    # get intensity features
                    intensity_speaker1 = librosa.feature.rms(y=wav_speaker_1, frame_length=int(sr*time_step), hop_length=int(sr*time_step), center=False)
                    intensity_speaker0 = librosa.feature.rms(y=wav_speaker_0, frame_length=int(sr*time_step), hop_length=int(sr*time_step), center=False)
                    # pad the speaker_array with zero at the front abd back
                    wav_speaker_0 = np.pad(wav_speaker_0, int(sr*time_step/2))
                    wav_speaker_1 = np.pad(wav_speaker_1, int(sr*time_step/2))
                    # get the mfcc features for speaker 0
                    winstep = int(math.floor(time_step * sr)) # number of samples per window
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
                    # store the files
                    torch.save(full_feat_speaker_0, "../data/processed_file/input/{}_part_{}_speaker_0.pt".format(annotation_speaker_name, s))
                    torch.save(full_feat_speaker_1, "../data/processed_file/input/{}_part_{}_speaker_1.pt".format(annotation_speaker_name, s))
                    torch.save(target_speaker_0, "../data/processed_file/target/{}_part_{}_speaker_0.pt".format(annotation_speaker_name, s))
                    torch.save(target_speaker_1, "../data/processed_file/target/{}_part_{}_speaker_1.pt".format(annotation_speaker_name, s))
                    # intensity input
                    torch.save(intensity_speaker1, "../data/processed_file/input/{}_part_{}_intensity_speaker_0.pt".format(annotation_speaker_name, s))
                    torch.save(intensity_speaker0, "../data/processed_file/input/{}_part_{}_intensity_speaker_1.pt".format(annotation_speaker_name, s))
                    # speech structure input
                    torch.save(turn_taking_feature_vector0, "../data/processed_file/input/{}_part_{}_turntaking_speaker_0.pt".format(annotation_speaker_name, s))
                    torch.save(turn_taking_feature_vector1, "../data/processed_file/input/{}_part_{}_turntaking_speaker_1.pt".format(annotation_speaker_name, s))
                    training_sample_list.append("{}_part_{}".format(annotation_speaker_name, s))
                with open("../data/processed_file/index.txt", "w") as f:
                    for line in range(0, len(training_sample_list)):
                        f.write(training_sample_list[line])
                        f.write("\n")
                    f.close()
                break
def prepare_data_correct(wav_dir, gaze_dir, length_of_training_audio_clips, sample_width=0.1, time_step=0.1, testing=False):
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
                try:
                    speaker_0_gaze_tier = grid['kijkrichting spreker1 [v] (TIE1)']
                    speaker_1_gaze_tier = grid['kijkrichting spreker2 [v] (TIE3)']
                    speaker_0_transcript_tier = grid['spreker1 [v] (TIE0)']
                    speaker_1_transcript_tier = grid['spreker2 [v] (TIE2)']
                except:
                    speaker_0_gaze_tier = grid['kijkrichting spreker1 (TIE1)']
                    speaker_1_gaze_tier = grid['kijkrichting spreker2 (TIE3)']
                    speaker_0_transcript_tier = grid['spreker1 [v] (TIE0)']
                    speaker_1_transcript_tier = grid['spreker2 [v] (TIE2)']
                # load audio
                audio, sr = librosa.load(os.path.join(wav_dir, audio_filepath), mono=False, sr=config["sample_rate"])
                # get the list of lower bound of the intervals
                annotation_speaker_0_starts = [i.xmin for i in speaker_0_gaze_tier]
                annotation_speaker_1_starts = [i.xmin for i in speaker_1_gaze_tier]
                transcript_speaker_0_starts = [i.xmin for i in speaker_0_transcript_tier]
                transcript_speaker_1_starts = [i.xmin for i in speaker_1_transcript_tier]
                
                # obtain total number of segments:
                end_time = speaker_1_gaze_tier[-1].xmax
                num_of_segments = math.floor(end_time / length_of_training_audio_clips)
                num_targets_per_segment = int(length_of_training_audio_clips / sample_width)
                num_samples_per_segment = int(sr * length_of_training_audio_clips)
                for s in range(num_of_segments):
                    for sp in range(2):
                        # get speaker_target
                        target_speaker = torch.zeros([num_targets_per_segment])
                        if sp == 0:
                            annotation_starts = annotation_speaker_0_starts
                            transcript_starts = transcript_speaker_0_starts
                            gaze_tier = speaker_0_gaze_tier
                            transcript_tier = speaker_0_transcript_tier
                        else:
                            annotation_starts = annotation_speaker_1_starts
                            transcript_starts = transcript_speaker_1_starts
                            gaze_tier = speaker_1_gaze_tier
                            transcript_tier = speaker_1_transcript_tier
                        #   speaking = 1, silent = 0
                        #   absolute time of current turn
                        #   normalized time since turn started
                        #   normalized time till turn end
                        #   time till next speech start
                        #   time till next pause start
                        #   time since last speech end
                        #   time since last pause end
                        turn_taking_feature_vector = torch.zeros([num_targets_per_segment, 8])
                        for sample in range(num_targets_per_segment):
                            time = (s * length_of_training_audio_clips) + (sample * sample_width)

                            index_gaze = bisect(annotation_starts, time)
                            target = 1 if gaze_tier[index_gaze - 1].text == 'g' else 0 # g is gaze, x is aversion
                            target_speaker[sample] = target
                            index_transcript = bisect(transcript_starts, time) - 1
                            # record whether the person is speaking (speaker 0)
                            if index_transcript >= 0:
                                if transcript_tier[index_transcript].text == "":
                                    speaking = 0 # feature 0
                                else:
                                    speaking = 1 # feature 0
                                start = transcript_tier[index_transcript].xmin
                                end = transcript_tier[index_transcript].xmax
                            else:
                                index_transcript = 0
                                speaking = 0
                                start = 0
                                end = transcript_tier[0].xmin
                            # record property of the current interval
                            interval_length = end - start # feature 1
                            time_since_start = time - start # feature 2
                            time_till_end = end - time # feature 3
                            time_till_next_speech = -1 # feature 4
                            for interval in range(index_transcript+1, len(transcript_tier)):
                                if transcript_tier[interval].text != "":
                                    time_till_next_speech = transcript_tier[interval].xmin - time
                                    break
                            if time_till_next_speech < 0:
                                time_till_next_speech = (end + 1) - time 
                            time_till_next_slience = -1 # feature 5
                            for interval in range(index_transcript+1, len(transcript_tier)):
                                if transcript_tier[interval].text == "":
                                    time_till_next_slience = transcript_tier[interval].xmin - time
                                    break
                            if time_till_next_slience < 0:
                                time_till_next_slience = (end + 1) - time 
                            time_till_prev_speech = -1 # feature 6
                            for interval in range(index_transcript-1, -1, -1):
                                if transcript_tier[interval].text != "":
                                    time_till_prev_speech = time - transcript_tier[interval].xmax
                                    break
                            if time_till_prev_speech < 0:
                                time_till_prev_speech = time - (start - 1)
                            time_till_prev_silence = -1 # feature 7
                            for interval in range(index_transcript-1, -1, -1):
                                if transcript_tier[interval].text == "":
                                    time_till_prev_silence = time - transcript_tier[interval].xmax
                                    break
                            if time_till_prev_silence < 0:
                                time_till_prev_silence = time - (start - 1)
                            turn_taking_feature_vector[sample, 0] = speaking
                            turn_taking_feature_vector[sample, 1] = interval_length
                            turn_taking_feature_vector[sample, 2] = time_since_start
                            turn_taking_feature_vector[sample, 3] = time_till_end
                            turn_taking_feature_vector[sample, 4] = time_till_next_speech
                            turn_taking_feature_vector[sample, 5] = time_till_next_slience
                            turn_taking_feature_vector[sample, 6] = time_till_prev_speech
                            turn_taking_feature_vector[sample, 7] = time_till_prev_silence
                        # get the wav files
                        wav_speaker = audio[sp, s * num_samples_per_segment: (s + 1) * num_samples_per_segment]
                        # get intensity features
                        intensity_speaker = librosa.feature.rms(y=wav_speaker, frame_length=int(sr*time_step), hop_length=int(sr*time_step), center=False)
                        # pad the speaker_array with zero at the front abd back
                        wav_speaker = np.pad(wav_speaker, int(sr*time_step/2))
                        # get the mfcc features for speaker 0
                        winstep = int(math.floor(time_step * sr)) # number of samples per window
                        mfcc_featspeaker = psf.mfcc(wav_speaker, samplerate=sr, winlen=config["window_length"],
                                            winstep=config["window_length"], nfft=winstep, numcep=13)
                        logfbank_featspeaker = psf.logfbank(wav_speaker, samplerate=sr, winlen=config["window_length"],
                                                    winstep=config["window_length"],nfft=winstep, nfilt=26)
                        ssc_featspeaker = psf.ssc(wav_speaker, samplerate=sr, winlen=config["window_length"],
                                        winstep=config["window_length"], nfft=winstep, nfilt=26)
                        full_feat_speaker = np.concatenate([mfcc_featspeaker, logfbank_featspeaker, ssc_featspeaker], axis=1)
                        torch.save(full_feat_speaker, "../data/processed_file/input/{}_part_{}_speaker_{}.pt".format(annotation_speaker_name, s, sp))
                        torch.save(target_speaker, "../data/processed_file/target/{}_part_{}_speaker_{}.pt".format(annotation_speaker_name, s, sp))
                        # intensity input
                        torch.save(intensity_speaker, "../data/processed_file/input/{}_part_{}_intensity_speaker_{}.pt".format(annotation_speaker_name, s, sp))
                        # speech structure input
                        torch.save(turn_taking_feature_vector, "../data/processed_file/input/{}_part_{}_turntaking_speaker_{}.pt".format(annotation_speaker_name, s, sp))
                        training_sample_list.append("{}_part_{}".format(annotation_speaker_name, s))
                with open("../data/processed_file/index.txt", "w") as f:
                    for line in range(0, len(training_sample_list)):
                        f.write(training_sample_list[line])
                        f.write("\n")
                    f.close()
                break
def prepare_data_positional_encoding(wav_dir, gaze_dir, length_of_training_audio_clips, sample_width=0.1, time_step=0.1, testing=False):
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
                try:
                    speaker_0_gaze_tier = grid['kijkrichting spreker1 [v] (TIE1)']
                    speaker_1_gaze_tier = grid['kijkrichting spreker2 [v] (TIE3)']
                    speaker_0_transcript_tier = grid['spreker1 [v] (TIE0)']
                    speaker_1_transcript_tier = grid['spreker2 [v] (TIE2)']
                except:
                    speaker_0_gaze_tier = grid['kijkrichting spreker1 (TIE1)']
                    speaker_1_gaze_tier = grid['kijkrichting spreker2 (TIE3)']
                    speaker_0_transcript_tier = grid['spreker1 [v] (TIE0)']
                    speaker_1_transcript_tier = grid['spreker2 [v] (TIE2)']
                # load audio
                audio, sr = librosa.load(os.path.join(wav_dir, audio_filepath), mono=False, sr=config["sample_rate"])
                # get the list of lower bound of the intervals
                annotation_speaker_0_starts = [i.xmin for i in speaker_0_gaze_tier]
                annotation_speaker_1_starts = [i.xmin for i in speaker_1_gaze_tier]
                transcript_speaker_0_starts = [i.xmin for i in speaker_0_transcript_tier]
                transcript_speaker_1_starts = [i.xmin for i in speaker_1_transcript_tier]
                
                # obtain total number of segments:
                end_time = speaker_1_gaze_tier[-1].xmax
                num_of_segments = math.floor(end_time / length_of_training_audio_clips)
                num_targets_per_segment = int(length_of_training_audio_clips / sample_width)
                num_samples_per_segment = int(sr * length_of_training_audio_clips)
                for s in range(num_of_segments):
                    for sp in range(2):
                        # get speaker_target
                        target_speaker = torch.zeros([num_targets_per_segment])
                        if sp == 0:
                            annotation_starts = annotation_speaker_0_starts
                            transcript_starts = transcript_speaker_0_starts
                            gaze_tier = speaker_0_gaze_tier
                            transcript_tier = speaker_0_transcript_tier
                        else:
                            annotation_starts = annotation_speaker_1_starts
                            transcript_starts = transcript_speaker_1_starts
                            gaze_tier = speaker_1_gaze_tier
                            transcript_tier = speaker_1_transcript_tier
                        #   speaking = 1, silent = 0
                        #   absolute time of current turn
                        #   normalized time since turn started
                        #   normalized time till turn end
                        #   time till next speech start
                        #   time till next pause start
                        #   time since last speech end
                        #   time since last pause end
                        turn_taking_feature_vector = torch.zeros([num_targets_per_segment, 3])
                        for sample in range(num_targets_per_segment):
                            time = (s * length_of_training_audio_clips) + (sample * sample_width)

                            index_gaze = bisect(annotation_starts, time)
                            target = 1 if gaze_tier[index_gaze - 1].text == 'g' else 0 # g is gaze, x is aversion
                            target_speaker[sample] = target
                            index_transcript = bisect(transcript_starts, time) - 1
                            # record whether the person is speaking (speaker 0)
                            if index_transcript >= 0:
                                if transcript_tier[index_transcript].text == "":
                                    speaking = 0 # feature 0
                                else:
                                    speaking = 1 # feature 0
                                start = transcript_tier[index_transcript].xmin
                                end = transcript_tier[index_transcript].xmax
                            else:
                                index_transcript = 0
                                speaking = 0
                                start = 0
                                end = transcript_tier[0].xmin
                            # record property of the current interval
                            interval_length = (end - start) # feature 1
                            turn_taking_feature_vector[sample, 0] = speaking
                            if interval_length > 0:
                                turn_taking_feature_vector[sample, 1] = interval_length/5
                                turn_taking_feature_vector[sample, 2] = (time - start)/interval_length # feature 2
                            else:
                                turn_taking_feature_vector[sample, 1] = 0
                                turn_taking_feature_vector[sample, 2] = 0
                            
                        # get the wav files
                        wav_speaker = audio[sp, s * num_samples_per_segment: (s + 1) * num_samples_per_segment]
                        # get intensity features
                        intensity_speaker = librosa.feature.rms(y=wav_speaker, frame_length=int(sr*time_step), hop_length=int(sr*time_step), center=False)
                        # pad the speaker_array with zero at the front abd back
                        wav_speaker = np.pad(wav_speaker, int(sr*time_step/2))
                        # get the mfcc features for speaker 0
                        winstep = int(math.floor(time_step * sr)) # number of samples per window
                        mfcc_featspeaker = psf.mfcc(wav_speaker, samplerate=sr, winlen=config["window_length"],
                                            winstep=config["window_length"], nfft=winstep, numcep=13)
                        logfbank_featspeaker = psf.logfbank(wav_speaker, samplerate=sr, winlen=config["window_length"],
                                                    winstep=config["window_length"],nfft=winstep, nfilt=26)
                        ssc_featspeaker = psf.ssc(wav_speaker, samplerate=sr, winlen=config["window_length"],
                                        winstep=config["window_length"], nfft=winstep, nfilt=26)
                        full_feat_speaker = np.concatenate([mfcc_featspeaker, logfbank_featspeaker, ssc_featspeaker], axis=1)
                        torch.save(full_feat_speaker, "../data/processed_file/input/{}_part_{}_speaker_{}.pt".format(annotation_speaker_name, s, sp))
                        torch.save(target_speaker, "../data/processed_file/target/{}_part_{}_speaker_{}.pt".format(annotation_speaker_name, s, sp))
                        # intensity input
                        torch.save(intensity_speaker, "../data/processed_file/input/{}_part_{}_intensity_speaker_{}.pt".format(annotation_speaker_name, s, sp))
                        # speech structure input
                        torch.save(turn_taking_feature_vector, "../data/processed_file/input/{}_part_{}_turntaking_speaker_{}.pt".format(annotation_speaker_name, s, sp))
                        training_sample_list.append("{}_part_{}".format(annotation_speaker_name, s))
                with open("../data/processed_file/index.txt", "w") as f:
                    for line in range(0, len(training_sample_list)):
                        f.write(training_sample_list[line])
                        f.write("\n")
                    f.close()
                break

class AudioDataset_Evan(torch.utils.data.Dataset):
    def __init__(self, config, data_dir, audio_length, window_length=0.01, time_step=0.01, listener_data=False):
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
        self.config = config
    def __len__(self):
        return len(self.data_paths) * 2
    def __getitem__(self, idx):
        channel = 0
        if idx >= len(self.data_paths):
            idx = idx // 2
            channel = 1
        file_name = self.data_paths[idx]
        input_file_path = os.path.join(self.data_dir, "input/{}_speaker_{}.pt".format(file_name, channel))
        input_sentence_structure_file_path = os.path.join(self.data_dir, "input/{}_turntaking_speaker_{}.pt".format(file_name, channel))
        input_intensity_file_path = os.path.join(self.data_dir, "input/{}_intensity_speaker_{}.pt".format(file_name, channel))
        target_file_path = os.path.join(self.data_dir, "target/{}_speaker_{}.pt".format(file_name, channel))
        if not self.listener_data:
            input_file_path = os.path.join(self.data_dir, "input/{}_speaker_{}.pt".format(file_name, channel))
            input_sentence_structure_file_path = os.path.join(self.data_dir, "input/{}_turntaking_speaker_{}.pt".format(file_name, channel))
            input_intensity_file_path = os.path.join(self.data_dir, "input/{}_intensity_speaker_{}.pt".format(file_name, channel))
            target_file_path = os.path.join(self.data_dir, "target/{}_speaker_{}.pt".format(file_name, channel))
            if self.config["input_modality"] == "intensity, sentence_structure":
                intensity_input = torch.FloatTensor(torch.load(input_intensity_file_path)).transpose(0, 1)
                sentence_modality_input = torch.FloatTensor(torch.load(input_sentence_structure_file_path))
                input = torch.cat([sentence_modality_input, intensity_input], dim=1)
                target = torch.FloatTensor(torch.load(target_file_path))
            elif self.config["input_modality"] == "mfcc":
                input = torch.FloatTensor(torch.load(input_file_path))
                target = torch.FloatTensor(torch.load(target_file_path))
        else:
            speaker = channel
            listener = 1 - speaker
            speaker_input_file_path = os.path.join(self.data_dir, "input/{}_speaker_{}.pt".format(file_name, speaker))
            speaker_input_sentence_structure_file_path = os.path.join(self.data_dir, "input/{}_turntaking_speaker_{}.pt".format(file_name, speaker))
            speaker_input_intensity_file_path = os.path.join(self.data_dir, "input/{}_intensity_speaker_{}.pt".format(file_name, speaker))
            target_file_path = os.path.join(self.data_dir, "target/{}_speaker_{}.pt".format(file_name, speaker))
            listener_input_file_path = os.path.join(self.data_dir, "input/{}_speaker_{}.pt".format(file_name, listener))
            listener_input_sentence_structure_file_path = os.path.join(self.data_dir, "input/{}_turntaking_speaker_{}.pt".format(file_name, listener))
            listener_input_intensity_file_path = os.path.join(self.data_dir, "input/{}_intensity_speaker_{}.pt".format(file_name, listener))
            if self.config["input_modality"] == "intensity, sentence_structure":
                speaker_intensity_input = torch.FloatTensor(torch.load(speaker_input_intensity_file_path)).transpose(0, 1)
                speaker_sentence_modality_input = torch.FloatTensor(torch.load(speaker_input_sentence_structure_file_path))
                listener_intensity_input = torch.FloatTensor(torch.load(listener_input_intensity_file_path)).transpose(0, 1)
                listener_sentence_modality_input = torch.FloatTensor(torch.load(listener_input_sentence_structure_file_path))
                input = torch.cat([speaker_intensity_input, listener_intensity_input, speaker_sentence_modality_input, listener_sentence_modality_input], dim=1)
                target = torch.FloatTensor(torch.load(target_file_path))
            elif self.config["input_modality"] == "mfcc":
                input = torch.FloatTensor(torch.load(speaker_input_file_path))
                target = torch.FloatTensor(torch.load(target_file_path))


        return input, target

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
    cold_start = False

    config = export_config_Evan()

    if cold_start:
        prepare_data_positional_encoding(wav_dir, gaze_dir,
                     config["sample_length"],
                     config["window_length"],
                     config["time_step"], testing=False)
    else:
        pass
    dataset = AudioDataset_Evan(config, data_dir, 5, 0.01, 0.01)
    data, target = dataset.__getitem__(0)