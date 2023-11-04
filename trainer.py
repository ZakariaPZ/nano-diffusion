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
                       batch_size):
        sampled_timesteps = torch.randint(0, self.ddpm.T, (batch_size,))
        t = self.ddpm.timesteps[sampled_timesteps]
        alpha_bar = self.ddpm.alpha_bar[sampled_timesteps]

        epsilon = torch.randn((batch_size, self.ddpm.dim, self.ddpm.dim, 3))

        # Unsqueeze to match dimensions of epsilon for broadcasting
        t = t[..., None, None, None]
        alpha_bar = alpha_bar[..., None, None, None]

        return t, alpha_bar, epsilon
    
    def run_epoch(self,
              batch_size,
              num_steps,
              x0,
              model):
        

        for batch_id, (data, label) in enumerate(train_loader):
            data = Variable(data)
            target = Variable(label)

        t, alpha_bar, epsilon = self.get_batch_data(batch_size)
        

        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * epsilon

        loss = self.loss(epsilon, model(xt))
        loss.backward()
        return loss.item()

    def train(self,
              num_steps,
              model):
        
        model.train() # because of dropout
        for i in range(num_steps):
            
