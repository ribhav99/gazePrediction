import torch
import os
import parselmouth

def get_participant_id_from_audio_clips(file_name):
    start_index = file_name.rfind('_') + 1
    end_index = file_name.index('.wav')
    id = file_name[start_index:end_index]
    channel_num = file_name[file_name.index('channel'):start_index-1]
    return id, channel_num


def sort_name_by_part_number(item):
    '''Used to traverse audio directory in order of audio clips'''
    end = item.rfind('_channel')
    start = len('_Number_')
    return int(item[start:end])


def weighted_binary_cross_entropy(output, target, weights=[57600/(2*21272), 57600/(2*36328)]):
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


def normalise_tensor(matrix):
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)


def find_path(file, folder):
  for f in os.listdir(folder):
    if f == file:
      return os.path.join(folder, file)
    if os.path.isdir(os.path.join(folder, f)):
      path = find_path(file, os.path.join(folder, f))
      if path:
        return path

def get_median_intensities(folder):
  intensities = {}
  for f in os.listdir(folder):
    path = os.path.join(folder, f)
    snd = parselmouth.Sound(path)
    intensity = torch.median(torch.tensor(snd.to_intensity(time_step=0.1).values).flatten())
    intensities[f] = intensity
  
  return intensities