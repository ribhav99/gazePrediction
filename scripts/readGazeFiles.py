from praatio import textgrid
from bisect import bisect

def get_intervals(file_path):
    tg = textgrid.openTextgrid(file_path, False)
    return tg.tierDict['kijkrichting spreker1 [v] (TIE1)'].entryList

def create_targets(file_path, audio_length=5, window_length=0.01):
    intervals = get_intervals(file_path)
    
if __name__ == '__main__':
    pass