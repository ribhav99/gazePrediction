import sys
sys.path.append('..')
import torch
import wandb
from tqdm import trange
from config.config import export_config # type: ignore

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
        wandb.init(project="gaze_prediction", config=config, save_code=True)
    else:
        wandb = None

    train(model, config, dataloader, wandb)