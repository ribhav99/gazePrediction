from praatio import textgrid
import os
import parselmouth

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
    snd = parselmouth.Sound('../data/wav_files_5_seconds/_Number_0_channel_0_DVA1A.wav')
    intensity = snd.to_intensity(time_step=0.1)
    pitch = snd.to_pitch(time_step=0.1)
    print(pitch.to_array())