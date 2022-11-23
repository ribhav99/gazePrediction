from vanillaCNN import CNNet
from createDataset import AudioDataset
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
import wandb
import utils
import yaml
import torch.nn as nn
from train import validation_confusion_matrix
import numpy as np
from config.config import export_config # type: ignore


wav_5_sec_dir = '../data/wav_files_5_seconds/'
gaze_dir = '../data/gaze_files'

run_obj = wandb.init(project="gaze_prediction", save_code=True,
                resume='allow', id='1j3g07a6')
config_file_name = 'config.yaml'
wandb.restore(config_file_name,
                        run_path=f'ribhav99/gaze_prediction/{run_obj.id}')
config_file_path = utils.find_path(config_file_name, 'wandb')



config = export_config()
'''
This works except the values that are nn.<something>
like the activation fn or pool. Those throw an error cause
the nn. is not at the start. If you can make it work
then the commented code here loads the confid from
wandb properly
'''
# config = {}
# with open(config_file_path, 'r') as f:
#     yaml_config = yaml.safe_load(f)
# for key in yaml_config:
#     try:
#         if 'value' in yaml_config[key]:
#             config[key] = yaml_config[key]['value']
#     except:
#         pass

all_data = AudioDataset(wav_5_sec_dir, gaze_dir, 5, config['window_length'],
        config['time_step'])
x, _ = all_data.__getitem__(0)
model = CNNet(config, [1] + list(x.shape), int(5/config["window_length"]))
valid_size = len(all_data) // 5
torch.manual_seed(6)
train, valid = torch.utils.data.random_split(all_data, [len(all_data) - valid_size, valid_size])
train_dataloader = torch.utils.data.DataLoader(train, config['batch_size'], True)
valid_dataloader = torch.utils.data.DataLoader(valid, config['batch_size'], True)

checkpoint_name = 'time=2022-11-22 08:26:12.786105_epoch=10.pt'
wandb.restore(checkpoint_name,
                        run_path=f'ribhav99/gaze_prediction/{run_obj.id}')
checkpoint_path = utils.find_path(checkpoint_name, 'wandb')

pretrained_dict = torch.load(checkpoint_path, map_location=config['device'])
model.load_weights(pretrained_dict)
model.to(config['device'])

validation_confusion_matrix(checkpoint_name, valid_dataloader, config, wandb, run_obj, model)

preds = []
targs = []
all_data_dataloader = torch.utils.data.DataLoader(all_data, batch_size=1, shuffle=True)
for batch, (x, y) in enumerate(all_data_dataloader):
    x = x.to(config['device'])
    pred = model(x)
    targs += y.detach().numpy().flatten().tolist()
    preds += pred.cpu().detach().numpy().flatten().tolist()

#     plt.plot(pred.cpu().detach().numpy().flatten(), label="prediction")
#     plt.plot(y.detach().numpy().flatten(), label="target")
#     plt.show()
# plt.plot(preds, label="prediction")
# plt.plot(targs, label="target")
# plt.show()

Plot, Axis = plt.subplots()
 
# Adjust the bottom size according to the
# requirement of the user
plt.subplots_adjust(bottom=0.25)
 
# Set the x_labels
x_labels = range(len(preds))
 
# plot the x and y using plot function
l = plt.plot(x_labels, preds, label="prediction")
l = plt.plot(x_labels, targs, label="target")
 
# Choose the Slider color
slider_color = 'White'
 
# Set the axis and slider position in the plot
axis_position = plt.axes([0.2, 0.1, 0.65, 0.03],
                         facecolor = slider_color)
slider_position = Slider(axis_position,
                         'Pos', 0.1, len(preds) - 100)
 
# update() function to change the graph when the
# slider is in use
def update(val):
    pos = slider_position.val
    Axis.axis([pos, pos+100, -0.25, 1.25])
    Plot.canvas.draw_idle()
 
# update function called using on_changed() function
slider_position.on_changed(update)
 
# Display the plot
plt.show()
