from vanillaCNN import CNNet
from createDataset import AudioDataset, load_single_audio_file_normalised
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
import wandb
import utils
import yaml
import torch.nn as nn
from train import validation_confusion_matrix


audio_file_path = '../data/wav_files_5_seconds/_Number_0_channel_0_DVB9N'
checkpoint_name = 'time=2022-11-29 06:22:13.018369_epoch=10.pt'
run_obj = wandb.init(project="gaze_prediction", save_code=True,
                resume='allow', id='288j7s6z')

config_file_name = 'config.yaml'
wandb.restore(config_file_name,
                        run_path=f'ribhav99/gaze_prediction/{run_obj.id}')
config_file_path = utils.find_path(config_file_name, 'wandb')



'''
This works but the loading config needs to be updated
if new functions are used or added, eg: pool, act_fn, etc
'''
config = {}
with open(config_file_path, 'r') as f:
    yaml_config = yaml.safe_load(f)
for key in yaml_config:
    try:
        if 'value' in yaml_config[key]:
            config[key] = yaml_config[key]['value']
    except:
        pass

config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
# activation_fn
if 'ReLU' in config["activation_fn"]:
    config["activation_fn"] = nn.ReLU()
# pool
start_ind = config['pool'].index('=') + 1
end_ind = config['pool'].index(',')
k_size = int(config['pool'][start_ind:end_ind])
if 'AvgPool2d' in config['pool']:
    config['pool'] = nn.AvgPool2d(k_size)
# loss_fn
if 'weighted_binary_cross_entropy' in config['loss_fn']:
    config['loss_fn'] = utils.weighted_binary_cross_entropy
'''
Finished Loading config
'''

x = load_single_audio_file_normalised(audio_file_path)
x = x.to(config['device'])
model = CNNet(config, [1] + list(x.shape), int(5/config["window_length"]))

wandb.restore(checkpoint_name,
                        run_path=f'ribhav99/gaze_prediction/{run_obj.id}')
checkpoint_path = utils.find_path(checkpoint_name, 'wandb')

pretrained_dict = torch.load(checkpoint_path, map_location=config['device'])
model.load_weights(pretrained_dict)
model.to(config['device'])

pred = model(x.unsqueeze(0))
pred = pred.cpu().detach().numpy().flatten().tolist()
print(pred)

Plot, Axis = plt.subplots()
 
# Adjust the bottom size
plt.subplots_adjust(bottom=0.25)
 
# Set the x_labels
x_labels = range(len(pred))
 
# plot the x and y using plot function
plt.plot(x_labels, pred, label="prediction")
plt.plot(x_labels, list(map(lambda x: 1 if round(x) == 1 else 0, pred)), label="rounded prediction")
 
# Choose the Slider color
slider_color = 'White'
 
# Set the axis and slider position in the plot
axis_position = plt.axes([0.2, 0.1, 0.65, 0.03],
                         facecolor = slider_color)
slider_position = Slider(axis_position,
                         'Pos', 0.1, len(pred) - 100)
 
# update() function to change the graph when the
# slider is in use
def update(val):
    pos = slider_position.val
    Axis.axis([pos, pos+100, -0.25, 1.25])
    Plot.canvas.draw_idle()
 
# update function called using on_changed() function
slider_position.on_changed(update)
 
# Display the plot
plt.legend(loc='best')
plt.show()
