import torch
from torch import nn

class Schedules:
    pass

class DDPM():

    def __init__(self, 
                 T,
                 dim) -> None:

        self.T = T 
        self.beta = self.get_linear_schedule(T)
        self.alpha_bar = torch.cumprod(1 - self.beta, dim=0)
        
        self.loss = nn.MSELoss()

    def get_linear_schedule(self, T):
    
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, T) # linear schedule

    def get_batch(self, shape):
        # Sample noise
        batch_size = shape[0]

        sampled_timesteps = torch.randint(0, self.T, (batch_size,))
        t = self.ddpm.timesteps[sampled_timesteps]
        epsilon = torch.randn_like(shape)

        # Unsqueeze to match dimensions of epsilon for broadcasting
        t = t[..., None, None, None]
        alpha_bar = alpha_bar[..., None, None, None]

        return t, alpha_bar, epsilon

    def forward_process(self,
                        x0,
                        t):
        '''
        DDPM forward process. Take image x0 and noise to timestep t.
        '''

        eps = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t]

        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps
        
        return xt

    def sample(self,
               model,
               shape=(1, 3, 64, 64)):
        '''
        DDPM reverse process. Algorithm 2 in the paper.
        '''
        model.eval()
        # Sample noise
        xt = torch.randn(shape)

        for t in torch.arange(0, self.T):
            t_batch = t.repeat(shape[0])
            alpha_t = 1 - self.beta[t_batch][..., None, None, None]
            alpha_bar = self.alpha_bar[t_batch][..., None, None, None]
            sigma_t = torch.sqrt(self.beta[t_batch])[..., None, None, None]

            if t > 1:
                z = torch.randn(shape)
            else:
                z = torch.zeros(shape)

            xtm1 = 1/torch.sqrt(alpha_t) * (xt - (1-alpha_t)/torch.sqrt(1-alpha_bar) * model(xt, t_batch)) + sigma_t * z
            xt = xtm1
        
        return xt 
    
    def train(self,
              x0,
              model,
              shape):
        '''
        Training algorithm for DDPM. Algorithm 1 in the paper.
        '''
        model.train()

        t, alpha_bar, epsilon = self.get_batch_data(shape)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * epsilon

        loss = self.loss(epsilon, model(xt, t))
        loss.backward()

        return loss
    