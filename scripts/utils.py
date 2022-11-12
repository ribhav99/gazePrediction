def get_participant_id_from_audio_clips(file_name):
    start_index = file_name.rfind('_') + 1
    end_index = file_name.index('.wav')
    return file_name[start_index:end_index]

def sort_name_by_part_number(item):
    '''Used to traverse audio directory in order of audio clips'''
    end = item.rfind('_')
    start = len('_Number_')
    return int(item[start:end])