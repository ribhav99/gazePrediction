import keyboard
import time

def annotate_video(video_path):
    annotation_arr = []
    while True:
        if keyboard.is_pressed('space'):
            annotation_arr.append(1)
        else:
            annotation_arr.append(0)

        if keyboard.is_pressed('q'):
            break
    
    print(annotation_arr)
if __name__ == '__main__':
    annotate_video(5)