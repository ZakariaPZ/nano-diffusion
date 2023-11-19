import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datasets import load_dataset 

from ddpm import DDPM
from schedulers import Scheduler
from model import MiniUNet

from tqdm import tqdm

def train(num_epochs,
          batch_size,
          n_timesteps,
          device):

    loss = nn.MSELoss()
    model = MiniUNet().to(device)
    scheduler = Scheduler(n_timesteps=n_timesteps).to(device)
    ddpm = DDPM(scheduler=scheduler, loss=loss).to(device)

    # Define the transformations to be applied to the images
    transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    def do_transforms(examples):
        examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
        del examples["image"]

        return examples

    dataset = load_dataset("fashion_mnist")
    transformed_dataset = dataset.with_transform(do_transforms).remove_columns("label")
    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

    model.train() 
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        ma_loss = 0
        with tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as t_bar:
            for batch_idx, data in enumerate(t_bar):

                x0 = data['pixel_values']
                optim.zero_grad()
                loss = ddpm.train(x0.to(device), model, x0.shape[0], x0.shape[1:])
                optim.step()

                ma_loss = (1 - (1/(batch_idx+1))) * ma_loss + (1/(batch_idx+1)) * loss
                t_bar.set_postfix(loss=ma_loss),
            
    torch.save(model, 'fashionmnist_ddpm.pth')

    # save model with custom path
    # model.eval()
    # validation 
    
if __name__ == '__main__':

    num_epochs = 100
    batch_size = 256
    n_timesteps = 1000
    device = 'cuda'

    train(num_epochs, batch_size, n_timesteps, device)
