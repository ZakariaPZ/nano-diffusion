import torch
import torch.nn as nn 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ddpm import DDPM, Scheduler
from model import MiniUNet

def train(num_epochs,
          batch_size,
          n_timesteps,
          device):

    model = MiniUNet(n_timesteps=n_timesteps).to(device)
    scheduler = Scheduler(n_timesteps=n_timesteps).to(device)
    ddpm = DDPM(scheduler=scheduler).to(device)

    # Define the transformations to be applied to the images
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the pixel values
    ])

    # Download the CIFAR-10 training dataset and apply the transformations
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Create a DataLoader to efficiently load and iterate through the dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
   
    model.train() 
    for epoch in range(num_epochs):
        loss = 0 
        for batch_idx, (x0, _) in enumerate(train_loader):
            loss += ddpm.train(x0.to(device), model, x0.shape)
        
        print(f'Loss in epoch {epoch+1}: {loss}')

    # save model 
    # model.eval()
    # validation 
    
if __name__ == '__main__':

    num_epochs = 2
    batch_size = 64    
    n_timesteps = 2
    device = 'cuda'

    train(num_epochs, batch_size, n_timesteps, device)
