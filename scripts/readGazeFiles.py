from cgitb import text
from praatio import textgrid
import textgrids
from bisect import bisect
import torch
import os

def get_intervals(file_path, channel):
    grid = textgrids.TextGrid(file_path)
    if channel == 0:
        try:
            return grid['kijkrichting spreker1 [v] (TIE1)']
        except:
            return grid['kijkrichting spreker1 (TIE1)']
    else:
        try:
            return grid['kijkrichting spreker2 [v] (TIE3)']        
        except:
            return grid['kijkrichting spreker2 (TIE3)']
def create_targets(file_path, channel, audio_length=5, window_length=0.1):
    # assert audio_length % window_length == 0
    intervals = get_intervals(file_path, channel)
    intervals_start = [i.xmin for i in intervals]
    num_clips = int(intervals[-1].xmax // audio_length)
    # num_clips = 180 # Bad practice but this is True
    num_windows_in_clip = int(audio_length / window_length)
    targets = torch.zeros([num_clips, num_windows_in_clip])

    for clip in range(num_clips):
        for window in range(num_windows_in_clip):
            time = (clip * audio_length) + (window * window_length)
            index =  bisect(intervals_start, time)
            # g is gaze, x is aversion
            target = 1 if intervals[index-1].text == 'g' else 0
            targets[clip][window] = target

    return targets

def create_targets_for_all_participants(folder_path, audio_length=5, window_length=0.1):
    all_targets = {}

    for file in os.listdir(folder_path):
        for channel in range(2):
            path = os.path.join(folder_path, file)
            participant_id = file[: file.index('.gaze')] + f'_channel_{channel}'
            target = create_targets(path, channel, audio_length, window_length)
            assert participant_id not in all_targets
            all_targets[participant_id] = target
        
    return all_targets
        

    
if __name__ == '__main__':
    torch.set_printoptions(profile="full")
    print(create_targets('../data/gaze_files/DVA1A.gaze', 0))
    # d = create_targets_for_all_participants('../data/gaze_files')
    # for i in d:
    #     print(d[i].shape)