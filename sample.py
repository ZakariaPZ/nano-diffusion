import torch
from torchvision.utils import save_image, make_grid
from samplers.ddpm import DDPM
from schedulers import Scheduler

scheduler = Scheduler(1000).to('cuda')
ddpm = DDPM(scheduler, torch.nn.MSELoss()).to('cuda')

model = torch.load('ddpm_fashion.pth')
model.eval()

xh = ddpm.sample(model, device='cuda', n_samples=4, shape=(1, 28, 28))
grid = make_grid(xh, normalize=True, value_range=(-1, 1), nrow=4)
save_image(grid, "sample.png")