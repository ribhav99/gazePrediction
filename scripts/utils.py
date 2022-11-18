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