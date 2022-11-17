import sys
sys.path.append('..')
import torch
import wandb
from tqdm import trange
import os
from config.config import export_config # type: ignore

def find_path(file, folder):
  for f in os.listdir(folder):
    if f == file:
      return os.path.join(folder, file)
    if os.path.isdir(os.path.join(folder, f)):
      path = find_path(file, os.path.join(folder, f))
      if path:
        return path

def train(model, config, dataloader, wandb):
    optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    device = config['device']
    loss_fn = config['loss_fn']
    model.to(device)
    model.train()

    for epoch in trange(config['epochs']):

        total_train_loss = 0

        for _, (X, Y) in enumerate(dataloader):

            X, Y = X.to(device), Y.to(device)
            optimiser.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimiser.step()

            del X, Y, pred
            torch.cuda.empty_cache()

            total_train_loss += loss.item()
        total_train_loss /= len(dataloader)

        if config['wandb']:
            wandb.log({'training loss': total_train_loss})
            
if __name__ == '__main__':
    from vanillaCNN import CNNet
    from createDataset import AudioDataset

    torch.manual_seed(6)
    wav_5_sec_dir = '../data/wav_files_5_seconds/'
    gaze_dir = '../data/gaze_files'
    config = export_config()
    model = CNNet(config)
    dataset = AudioDataset(wav_5_sec_dir, gaze_dir, 5, 0.1)
    dataloader = torch.utils.data.DataLoader(dataset,
                    batch_size=config['batch_size'],
                    shuffle=True)
    
    if config['wandb']:
        wandb.login()
        if config["load_model"]:
            wandb.init(project="gaze_prediction", config=config, save_code=True, resume='allow', id='1frbu4kq')
            checkpoint_name = 'time=2022-11-16 18:13:12.586469_epoch=11.pt'
            wandb.restore(checkpoint_name,
                                    run_path='ribhav99/gaze_prediction/30utm6c5')
            checkpoint_path = find_path(checkpoint_name, '/content/wandb')
            pretrained_dict = torch.load(checkpoint_path, map_location=config['device'])
            model.load_weights(pretrained_dict)
            model.to(config['device'])
        else:
            wandb.init(project="gaze_prediction", config=config, save_code=True)
    else:
        wandb = None

    train(model, config, dataloader, wandb)