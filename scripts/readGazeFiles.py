from praatio import textgrid
from bisect import bisect
import torch
import os

def get_intervals(file_path):
    tg = textgrid.openTextgrid(file_path, False)
    return tg.tierDict['kijkrichting spreker1 [v] (TIE1)'].entryList
    

def create_targets(file_path, audio_length=5, window_length=0.1):
    # assert audio_length % window_length == 0
    intervals = get_intervals(file_path)
    intervals_start = [i.start for i in intervals]
    # num_clips = int(intervals[-1].end // audio_length)
    num_clips = 180 # Bad practice but this is True
    num_windows_in_clip = int(audio_length / window_length)
    targets = torch.zeros([num_clips, num_windows_in_clip])

    for clip in range(num_clips):
        for window in range(num_windows_in_clip):
            time = (clip * audio_length) + (window * window_length)
            index =  bisect(intervals_start, time)
            target = 1 if intervals[index-1].label == 'g' else 0
            targets[clip][window] = target

    return targets

def create_targets_for_all_participants(folder_path, audio_length=5, window_length=0.1):
    all_targets = {}

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        participant_id = file[: file.index('.gaze')]
        target = create_targets(path, audio_length, window_length)
        assert participant_id not in all_targets
        all_targets[participant_id] = target
    
    return all_targets
        

    
if __name__ == '__main__':
    print(create_targets('../data/gaze_files/DVA1A.gaze'))