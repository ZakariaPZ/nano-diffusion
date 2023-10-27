import torch
import torch.nn as nn 


class Trainer:

    def __init__(self,
                 ddpm) -> None:
        self.ddpm = ddpm
        self.loss = nn.MSELoss()
    
    def train(self,
              batch_size,
              x0,
              model):
        
        sampled_timesteps = torch.randint(0, self.ddpm.T, (batch_size,))
        t = self.ddpm.timesteps[sampled_timesteps]
        alpha_bar = self.ddpm.alpha_bar[sampled_timesteps]

        epsilon = torch.randn((batch_size, self.ddpm.dim, self.ddpm.dim, 3))
        
        # Unsqueeze to match dimensions of epsilon for broadcasting
        t = t[..., None, None, None]
        alpha_bar = alpha_bar[..., None, None, None]

        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * epsilon

        loss = self.loss(epsilon, model(xt))
        loss.backward()
        return loss.item()