import sys
sys.path.append('..')
import torch
from torchmetrics.classification import BinaryF1Score, F1Score
import wandb
from tqdm import trange
import os
from datetime import datetime
from sklearn import metrics
import utils
from matplotlib import pyplot as plt
from config.config import export_config # type: ignore
from config.config import export_config_Evan # type: ignore

from LSTM_models import *
from Preprocessing_CreateDataset import AudioDataset_Evan


def train_model(model, config, train_data, valid_data, wandb):
    optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    device = config['device']
    loss_fn = config['loss_fn']
    model.to(device)
    model.train()

    training_loss = []
    valid_loss = []
    training_f1 = []
    valid_f1 = []
    count = 0

    # f1_score = BinaryF1Score(num_classes=2).to(device)
    f1_score = F1Score(task="multiclass", num_classes=2, average="weighted").to(device)
    for epoch in trange(1, config['epochs'] + 1):
        total_train_loss = 0
        total_valid_loss = 0
        total_train_f1 = 0
        total_valid_f1 = 0
        train_batch_counter = 0
        valid_batch_counter = 0
        model.zero_grad()
        for _, (X, Y) in enumerate(train_data):
            train_batch_counter += 1
            X, Y = X.to(device), Y.to(device)
            optimiser.zero_grad()
            if "Transformer" in config["model_type"]:
                pred = model(X, Y)
            else:
                pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimiser.step()
            total_train_loss += loss.item()

            binary_pred = torch.round(pred)
            f1_train = f1_score(binary_pred, Y).item()
            total_train_f1 += f1_train
            del X, Y, pred
            torch.cuda.empty_cache()

        total_train_f1 /= train_batch_counter
        total_train_loss /= len(train_data)

        for _, (X, Y) in enumerate(valid_data):
            with torch.no_grad():
                valid_batch_counter += 1
                X, Y = X.to(device), Y.to(device)
                if "Transformer" in config["model_type"]:
                    all_zero = torch.zeros(Y.shape).to(device)
                    pred = model(X, all_zero)
                else:
                    pred = model(X)
                loss = loss_fn(pred, Y)
                total_valid_loss += loss.item()

                binary_pred = torch.round(pred)
                f1_valid = f1_score(binary_pred, Y).item()
                total_valid_f1 += f1_valid
                
                del X, Y, pred
                torch.cuda.empty_cache()

        total_valid_f1 /= valid_batch_counter
        total_valid_loss /= len(valid_data)

        if config['wandb']:
            wandb.log({'training loss': total_train_loss,
                        'validation_loss': total_valid_loss,
                        'training_f1': total_train_f1,
                        'validation_f1': total_valid_f1})
        training_loss.append(total_train_loss)
        valid_loss.append(total_valid_loss)
        training_f1.append(total_train_f1)
        valid_f1.append(total_valid_f1)

        if total_valid_loss == min(valid_loss):
            file_name = f'time={datetime.now()}_epoch={epoch}.pt'
            torch.save(model.state_dict(), os.path.join("wandb/garb/", file_name))
                
                
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
        wandb.save(os.path.join("wandb/garb/", file_name))
    return file_name
def validation_confusion_matrix(model_name, valid_data, config, wandb, run_obj, model):
    if config['wandb']:
        wandb.restore(model_name, run_path=f'ribhav99/gaze_prediction/{run_obj.id}')
        model_path = utils.find_path(model_name, 'wandb')
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
    torch.manual_seed(6)
    config = export_config_Evan()
    all_data = AudioDataset_Evan(config, "../data/processed_file", config["sample_length"], config["window_length"], config["time_step"], config["use_listener"])
    x, _ = all_data.__getitem__(0)
    # load model:
    model = load_model(config)
    
    valid_size = len(all_data) // 5
    torch.manual_seed(6)
    train, valid = torch.utils.data.random_split(all_data, [len(all_data) - valid_size, valid_size])
    train_dataloader = torch.utils.data.DataLoader(train, config['batch_size'], True)
    valid_dataloader = torch.utils.data.DataLoader(valid, config['batch_size'], True)

    if config['wandb']:
        wandb.login()
        if config["load_model"]:
            run_obj = wandb.init(project="gaze_prediction", config=config, save_code=True,
                resume='allow', id='1frbu4kq')
            checkpoint_name = 'time=2022-11-16 18:13:12.586469_epoch=11.pt'
            wandb.restore(checkpoint_name,
                                    run_path=f'ribhav99/gaze_prediction/{run_obj.id}')
            checkpoint_path = utils.find_path(checkpoint_name, 'wandb')
            pretrained_dict = torch.load(checkpoint_path, map_location=config['device'])
            model.load_weights(pretrained_dict)
            model.to(config['device'])
        else:
            run_obj = wandb.init(project="gaze_prediction", config=config, save_code=True)
    else:
        wandb = None

    
    best_model_name = train_model(model, config, train_dataloader, valid_dataloader, wandb)
    validation_confusion_matrix(best_model_name, valid_dataloader, config, wandb, run_obj, model)
    for f in os.listdir('.'):
        if f.endswith('.pt') and f != best_model_name:
            os.remove(f)