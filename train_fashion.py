import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import load_dataset 
from PIL import Image
import numpy as np

from samplers.ddpm import DDPM
from schedulers import Scheduler
from model import MiniUNet

from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8))
            x = self.transform(x)
        
        return x
    
    def __len__(self):
        return len(self.data)


def train(num_epochs,
          batch_size,
          n_timesteps,
          device):

    loss = nn.MSELoss()
    model = MiniUNet().to(device)
    scheduler = Scheduler(n_timesteps=n_timesteps).to(device)
    ddpm = DDPM(scheduler=scheduler, loss=loss).to(device)

    train_data = np.load('data/x_train.npy')
    val_data = np.load('data/x_val.npy')

    # Define the transformations to be applied to the images
    transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    train_dataset = MyDataset(train_data, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = MyDataset(val_data, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optim, 'min', factor=0.1, patience=10, verbose=True)

    for epoch in range(num_epochs):

        model.train() 
        ma_loss = 0
        
        with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as t_bar:
            for batch_idx, x0 in enumerate(t_bar):

                optim.zero_grad()
                loss = ddpm.train(x0.to(device), model, x0.shape[0], x0.shape[1:])
                loss.backward()
                optim.step()

                ma_loss = (1 - (1/(batch_idx+1))) * ma_loss + (1/(batch_idx+1)) * loss.item()
                t_bar.set_postfix(loss=ma_loss)
        
        if epoch % 20 == 0:
            torch.save(model, f'run/ddpm_fashion_{epoch + 1}.pth')

        # validation
        model.eval()
        val_ma_loss = 0
        with torch.no_grad():
            with tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as t_bar:
                for batch_idx, x0 in enumerate(t_bar):

                    loss = ddpm.train(x0.to(device), model, x0.shape[0], x0.shape[1:])
                    val_ma_loss = (1 - (1/(batch_idx+1))) * val_ma_loss + (1/(batch_idx+1)) * loss.item()
                    t_bar.set_postfix(loss=val_ma_loss)
    
        scheduler.step(val_ma_loss)

    torch.save(model, 'ddpm_fashion.pth')
    # save model with custom path
    # model.eval()
    # validation 
    
if __name__ == '__main__':

    num_epochs = 200
    batch_size = 256
    n_timesteps = 1000
    device = 'cuda'

    train(num_epochs, batch_size, n_timesteps, device)
