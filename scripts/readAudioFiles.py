import wave
import numpy as np
import os
from tqdm import tqdm

def save_wav_channel(save_path, load_path, channel):
    wav = wave.open(load_path, 'rb')
    nch   = wav.getnchannels()
    depth = wav.getsampwidth()
    wav.setpos(0)
    sdata = wav.readframes(wav.getnframes())

    # Extract channel data (24-bit data not supported)
    typ = { 1: np.uint8, 2: np.uint16, 4: np.uint32 }.get(depth)
    if not typ:
        raise ValueError("sample width {} not supported".format(depth))
    if channel >= nch:
        raise ValueError("cannot extract channel {} out of {}".format(channel+1, nch))
    # print ("Extracting channel {} out of {} channels, {}-bit depth".format(channel+1, nch, depth*8))
    data = np.fromstring(sdata, dtype=typ)
    ch_data = data[channel::nch]

    # Save channel to a separate file
    outwav = wave.open(save_path, 'w')
    outwav.setparams(wav.getparams())
    outwav.setnchannels(1)
    outwav.writeframes(ch_data.tostring())
    outwav.close()
    wav.close()


def create_audio_data(file_path, save_folder, audio_length=5):
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    wav = wave.open(file_path, 'rb')
    length = wav.getnframes() / wav.getframerate()
    frames_per_second_for_reading = wav.getframerate() * wav.getsampwidth()
    frames = wav.readframes(-1)

    # print(frames_per_second_for_reading * length, len(wav.readframes(-1)))
    for i in range(int(length/audio_length)):
        file_name = os.path.basename(file_path)
        save_path = os.path.join(save_folder, f'_Number_{i}_{file_name}')
        outwav = wave.open(save_path, 'wb')
        outwav.setparams(wav.getparams())
        outwav.setnframes(frames_per_second_for_reading * audio_length)
        outwav.writeframes(frames[frames_per_second_for_reading * audio_length*i:frames_per_second_for_reading * audio_length*(i+1)])
        outwav.close()
    wav.close()


def convert_to_mono_channel(save_folder, load_folder):
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    for file in tqdm(os.listdir(load_folder)):
        file_path = os.path.join(load_folder, file)
        save_path = os.path.join(save_folder, file)
        save_wav_channel(save_path, file_path, 0)

if __name__ == '__main__':
    wav_folder = '../data/wav_files_single_channel/'
    convert_to_mono_channel(wav_folder, '../data/wav_files')
    for file in tqdm(os.listdir(wav_folder)):
        path = os.path.join(wav_folder, file)
        create_audio_data(path, '../data/wav_files_5_seconds')
    