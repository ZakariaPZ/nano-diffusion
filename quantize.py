import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from model import MiniUNet
from samplers.ddpm import DDPM
from schedulers import Scheduler
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

model = torch.load('ddpm_fashion.pth')
model.eval()

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters in the model:", num_params)

# Each parameter is a 32-bit float
n_bits = num_params * 32

n_mb = n_bits / (8 * 1024 ** 2)
print("Number of megabytes:", n_mb)



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


def test():

    device = 'cuda'
    n_timesteps = 100
    batch_size = 256
    

    # Define the transformations to be applied to the images
    transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
    ])


    test_data = np.load('data/x_test.npy')
    test_dataset = MyDataset(test_data, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = MiniUNet(quantize=True).to(device)
    model.load_state_dict(torch.load('run/ddpm_fashion_181.pth'))

    scheduler = Scheduler(n_timesteps=n_timesteps).to(device)
    ddpm = DDPM(scheduler, torch.nn.MSELoss()).to('cuda')

    model.eval()
    val_ma_loss = 0
    with torch.no_grad():
        with tqdm(test_dataloader, desc=f'Test Evaluation', unit='batch') as t_bar:
            for batch_idx, x0 in enumerate(t_bar):

                loss = ddpm.train(x0.to(device), model, x0.shape[0], x0.shape[1:])
                test_ma_loss = (1 - (1/(batch_idx+1))) * val_ma_loss + (1/(batch_idx+1)) * loss.item()
                t_bar.set_postfix(loss=test_ma_loss)

test()
# param_size = 0
# for param in model.parameters():
#     param_size += param.nelement() * param.element_size()
# buffer_size = 0
# for buffer in model.buffers():
#     buffer_size += buffer.nelement() * buffer.element_size()

# size_all_mb = (param_size + buffer_size) / 1024**2
# print('model size: {:.3f}MB'.format(size_all_mb))