import sys
sys.path.append('..')
import torch
import wandb
from tqdm import trange
import os
from datetime import datetime
from sklearn import metrics
import shutil
from config.config import export_config # type: ignore

def find_path(file, folder):
  for f in os.listdir(folder):
    if f == file:
      return os.path.join(folder, file)
    if os.path.isdir(os.path.join(folder, f)):
      path = find_path(file, os.path.join(folder, f))
      if path:
        return path

def train_model(model, config, train_data, valid_data, wandb):
    optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    device = config['device']
    loss_fn = config['loss_fn']
    model.to(device)
    model.train()

    training_loss = []
    valid_loss = []
    count = 0

    for epoch in trange(1, config['epochs'] + 1):

        total_train_loss = 0
        total_valid_loss = 0

        for _, (X, Y) in enumerate(train_data):

            X, Y = X.to(device), Y.to(device)
            optimiser.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimiser.step()
            total_train_loss += loss.item()

            del X, Y, pred
            torch.cuda.empty_cache()

            
        total_train_loss /= len(train_data)

        for _, (X, Y) in enumerate(valid_data):
            with torch.no_grad():
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                loss = loss_fn(pred, Y)
                total_valid_loss += loss.item()

                del X, Y, pred
                torch.cuda.empty_cache()

                
        total_valid_loss /= len(valid_data)

        if config['wandb']:
            wandb.log({'training loss': total_train_loss,
                        'validation_loss': total_valid_loss})
        
        training_loss.append(total_train_loss)
        valid_loss.append(total_valid_loss)

        if total_valid_loss == min(valid_loss):
            file_name = f'time={datetime.now()}_epoch={epoch}.pt'
            torch.save(model.state_dict(), file_name)
                
                
        if config['early_stopping']:
            if epoch > 1:
                if total_valid_loss > valid_loss[epoch - 2]:
                    count += 1
                else:
                    count = 0

            if count == config['early_stopping']:
                print('\n\nStopping early due to decrease in performance on validation set\n\n')
                break 
        
    if config['wandb']:
        wandb.save(file_name)
    return file_name  

def validation_confusion_matrix(model_path, valid_data, config, wandb, run_obj):
    if config['wandb']:
        wandb.restore(model_path, run_path=f'ribhav99/gaze_prediction/{run_obj.id}')
        model_path = find_path(model_path, 'wandb')
    pretrained_dict = torch.load(model_path, map_location=config['device'])
    model.load_weights(pretrained_dict)
    model.to(config['device'])
    model.eval()
    device = config['device']

    all_targets = []
    all_predictions = []
    for _, (X, Y) in enumerate(valid_data):
        with torch.no_grad():
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            # for t in Y:
            #     all_targets += [i for i in t]
            # for o in pred:
            #     all_predictions += [torch.round(i) for i in o]
            all_targets += torch.flatten(Y).cpu()
            all_predictions += torch.flatten(pred).cpu()

            del X, Y, pred
            torch.cuda.empty_cache()

    all_predictions = [torch.round(i) for i in all_predictions]
    cm = metrics.confusion_matrix(all_targets, all_predictions)
    mets = metrics.classification_report(all_targets, all_predictions)
    print(mets)
    print(cm)


if __name__ == '__main__':
    from vanillaCNN import CNNet
    from createDataset import AudioDataset

    torch.manual_seed(6)
    wav_5_sec_dir = '../data/wav_files_5_seconds/'
    gaze_dir = '../data/gaze_files'
    config = export_config()
    model = CNNet(config)
    all_data = AudioDataset(wav_5_sec_dir, gaze_dir, 5, 0.1)
    valid_size = len(all_data) // 5
    torch.manual_seed(6)
    train, valid = torch.utils.data.random_split(all_data, [len(all_data) - valid_size, valid_size])
    train_dataloader = torch.utils.data.DataLoader(train, config['batch_size'], True)
    valid_dataloader = torch.utils.data.DataLoader(valid, config['batch_size'], True)
    if config['wandb']:
        wandb.login()
        if config["load_model"]:
            run_obj = wandb.init(project="gaze_prediction", config=config, save_code=True, resume='allow', id='1frbu4kq')
            checkpoint_name = 'time=2022-11-16 18:13:12.586469_epoch=11.pt'
            wandb.restore(checkpoint_name,
                                    run_path=f'ribhav99/gaze_prediction/{run_obj.id}')
            checkpoint_path = find_path(checkpoint_name, '/content/wandb')
            pretrained_dict = torch.load(checkpoint_path, map_location=config['device'])
            model.load_weights(pretrained_dict)
            model.to(config['device'])
        else:
            run_obj = wandb.init(project="gaze_prediction", config=config, save_code=True)
    else:
        wandb = None

    
    best_model_name = train_model(model, config, train_dataloader, valid_dataloader, wandb)
    validation_confusion_matrix(best_model_name, valid_dataloader, config, wandb, run_obj)
    for f in os.listdir('.'):
        if f.endswith('.pt'):
            os.remove(f)
    shutil.rmtree('wandb')
