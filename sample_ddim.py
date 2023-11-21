import torch
from torchvision.utils import save_image, make_grid
from samplers.ddim import DDIM
from schedulers import Scheduler

scheduler = Scheduler(1000).to('cuda')
ddim = DDIM(scheduler, torch.nn.MSELoss()).to('cuda')

model = torch.load('ddpm_fashion.pth')
model.eval()

xh = ddim.sample(model, device='cuda', n_samples=1, shape=(1, 28, 28), n_inference_steps=1000)
print(xh.shape)
grid = make_grid(xh, normalize=True, value_range=(-1, 1), nrow=1)
save_image(grid, "sample.png")