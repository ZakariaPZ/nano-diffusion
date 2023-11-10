import torch
import torch.nn as nn 


class Trainer:

    def __init__(self,
                 ddpm,
                 train_loader) -> None:
        self.ddpm = ddpm
        self.train_loader = train_loader
        self.loss = nn.MSELoss()

    def get_batch_data(self,
                       batch_size,
                       num_channels=3):
        sampled_timesteps = torch.randint(0, self.ddpm.T, (batch_size,))
        t = self.ddpm.timesteps[sampled_timesteps]
        alpha_bar = self.ddpm.alpha_bar[sampled_timesteps]

        epsilon = torch.randn((batch_size, num_channels, self.ddpm.dim, self.ddpm.dim))

        # Unsqueeze to match dimensions of epsilon for broadcasting
        t = t[..., None, None, None]
        alpha_bar = alpha_bar[..., None, None, None]

        return t, alpha_bar, epsilon
    
    def run_epoch(self,
              batch_size,
              num_steps,
              x0,
              model):
        

        for batch_id, (x0, _) in enumerate(self.train_loader):
            t, alpha_bar, epsilon = self.get_batch_data(batch_size, num_channels=x0.shape[1])
            xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * epsilon

            loss = self.loss(epsilon, model(xt))
            loss.backward()
            return loss.item()

        loss = self.loss(epsilon, model(xt))
        loss.backward()
        return loss.item()

    def train(self,
              num_steps,
              model):
        
        model.train() # because of dropout
        for i in range(num_steps):
            break
