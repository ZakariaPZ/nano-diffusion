import torch
from torch import nn

class Scheduler(nn.Module):
    def __init__(self,
                 n_timesteps):
        super(Scheduler, self).__init__()
        # TODO: Register class members to buffer to make device transfer easier
        self.n_timesteps = n_timesteps
        self.register_buffer('beta', self.linear_schedule())
        self.register_buffer('alpha_bar', torch.cumprod(1 - self.beta, dim=0))

    def linear_schedule(self):
    
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, self.n_timesteps) # linear schedule

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
    