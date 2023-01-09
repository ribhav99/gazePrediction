import torch 
from matplotlib import pyplot as plt
from Preprocessing_CreateDataset import AudioDataset_Evan
import os
from torchmetrics.classification import BinaryF1Score
from LSTM_models import load_model, LSTM_BN
from config.config import export_config_Evan
from train import validation_confusion_matrix
from sklearn import metrics

def dataset_stats():
    total_aversion = 0
    total_gaze = 0
    data_set_location = "../data/processed_file"
    set = AudioDataset_Evan(data_set_location, 5, 0.01, 0.01)
    f1score = BinaryF1Score()
    total_ts = 0
    label_0 = 0
    label_1 = 0
    # compute mean f1 score
    f1_scores_if_all_1 = 0
    f1_scores_if_all_0 = 0
    counter = 0
    for i in range(len(set)):
        _, gaze_label = set[i]
        total_ts += gaze_label.shape[0]
        label_1 += gaze_label.sum().item()
        label_0 += gaze_label.shape[0] - gaze_label.sum().item()
        ones = torch.ones(gaze_label.shape)
        zeros = torch.zeros(gaze_label.shape)
        f1_scores_if_all_1 += f1score(ones, gaze_label).item()
        f1_scores_if_all_0 += f1score(zeros, gaze_label).item()
        counter += 1
    mean_f1_score_if_all_1 = f1_scores_if_all_1 / counter
    mean_f1_score_if_all_0 = f1_scores_if_all_0 / counter
    
    print("the percentage of gaze is: {}".format(label_1 / total_ts))
    print("the average f1 score if predicting all 1 is {}".format(mean_f1_score_if_all_1))
    print("the average f1 score if predicting all 0 is {}".format(mean_f1_score_if_all_0))
    
if __name__ == "__main__":
    config = export_config_Evan()
    model = load_model(config)
    model.eval()

    data_set_location = "../data/processed_file"
    all_data = AudioDataset_Evan(data_set_location, 5, 0.01, 0.01)
    valid_size = len(all_data) // 5
    torch.manual_seed(6)
    train, valid = torch.utils.data.random_split(all_data, [len(all_data) - valid_size, valid_size])
    train_dataloader = torch.utils.data.DataLoader(train, config['batch_size'], True)
    valid_dataloader = torch.utils.data.DataLoader(valid, config['batch_size'], True)

    device = config['device']

    all_targets = []
    all_predictions = []
    for _, (X, Y) in enumerate(valid_dataloader):
        with torch.no_grad():
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            all_targets += torch.flatten(Y).cpu()
            all_predictions += torch.flatten(pred).cpu()
            del X, Y, pred
            torch.cuda.empty_cache()

    all_predictions = [torch.round(i) for i in all_predictions]
    cm = metrics.confusion_matrix(all_targets, all_predictions)
    metrics.precision_recall_fscore_support(all_predictions, all_targets)
    mets = metrics.classification_report(all_targets, all_predictions)
    print(mets)
    print(cm)

    