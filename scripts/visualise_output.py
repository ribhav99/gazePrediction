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

if __name__ == "__main__":
    # wandb.login(key="6c05ad41b6f62f0b7aad8fe4074fede07526eced")
    wav_5_sec_dir = '../data/wav_files_5_seconds/'
    gaze_dir = '../data/gaze_files'
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

    all_data = AudioDataset(wav_5_sec_dir, gaze_dir, 5, config['window_length'],
            config['time_step'])
    x, _ = all_data.__getitem__(0)
    model = CNNet(config, [1] + list(x.shape), int(5/config["window_length"]))
    valid_size = len(all_data) // 5
    torch.manual_seed(6)
    train, valid = torch.utils.data.random_split(all_data, [len(all_data) - valid_size, valid_size])
    train_dataloader = torch.utils.data.DataLoader(train, config['batch_size'], True)
    valid_dataloader = torch.utils.data.DataLoader(valid, config['batch_size'], True)

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

    Plot, Axis = plt.subplots()

    # Adjust the bottom size
    plt.subplots_adjust(bottom=0.25)

    # Set the x_labels
    x_labels = range(len(preds))

    # plot the x and y using plot function
    plt.plot(x_labels, preds, label="prediction")
    plt.plot(x_labels, targs, label="target")
    plt.plot(x_labels, list(map(lambda x: 0.9 if round(x) == 1 else 0.1, preds)), label="rounded prediction")

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
    plt.legend(loc='best')
    plt.show()
