from praatio import textgrid
import os

def get_intervals(file_path):
    tg = textgrid.openTextgrid(file_path, False)
    return tg.tierDict['kijkrichting spreker1 [v] (TIE1)'].entryList


if __name__ == '__main__':
    folder_path = '../data/gaze_files'
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        try:
            get_intervals(path)
        except:
            print(file)