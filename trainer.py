import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ddpm import DDPM, Scheduler
from model import MiniUNet

from tqdm import tqdm

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
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as t_bar:
            for batch_idx, (x0, _) in enumerate(t_bar):


                loss += ddpm.train(x0.to(device), model, x0.shape[0], x0.shape[1:])
            
            print(f'Loss in epoch {epoch+1}: {loss}')

    torch.save(model, 'cifar10_ddpm.pth')

    # save model with custom path
    # model.eval()
    # validation 
    
if __name__ == '__main__':

    num_epochs = 10
    batch_size = 64    
    n_timesteps = 1000
    device = 'cuda'

    train(num_epochs, batch_size, n_timesteps, device)
