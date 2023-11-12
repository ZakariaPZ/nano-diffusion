import torch
from torch import nn

class Scheduler(nn.Module):
    def __init__(self,
                 n_timesteps):
        super(Scheduler, self).__init__()
        self.n_timesteps = n_timesteps
        self.beta = self.linear_schedule()
        self.alpha_bar = torch.cumprod(1 - self.beta, dim=0)

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
    

class DDPM(nn.Module):

    def __init__(self, 
                 scheduler) -> None:
        super(DDPM, self).__init__()

        self.scheduler = scheduler
        self.loss = nn.MSELoss()

    def sample(self,
               model,
               device,
               n_samples,
               shape=(3, 64, 64)):
        '''
        DDPM reverse process. Algorithm 2 in the paper.
        '''
        model.eval()
        # Sample noise
        xt = torch.randn(n_samples, *shape).to(device)

        for t in torch.arange(0, self.scheduler.T):
            t_batch = t.repeat(n_samples).to(device)
            alpha_t = 1 - self.scheduler.beta[t_batch][..., None, None, None]
            alpha_bar = self.scheduler.alpha_bar[t_batch][..., None, None, None]
            sigma_t = torch.sqrt(self.scheduler.beta[t_batch])[..., None, None, None]

            if t > 1:
                z = torch.randn(n_samples, *shape).to(device)
            else:
                z = torch.zeros(n_samples, *shape).to(device)

            xtm1 = 1/torch.sqrt(alpha_t) * (xt - (1-alpha_t)/torch.sqrt(1-alpha_bar) * model(xt, t_batch)) + sigma_t * z
            xt = xtm1
        
        return xt 
    
    def train(self,
              x0,
              model,
              batch_size,
              shape):
        '''
        Training algorithm for DDPM. Algorithm 1 in the paper.
        '''
        model.train()

        t = torch.randint(0, self.scheduler.n_timesteps, (batch_size,)).to(x0.device)
        alpha_bar = self.scheduler.alpha_bar[t]
        epsilon = torch.randn(batch_size, *shape).to(x0.device)

        # Unsqueeze to match dimensions of epsilon for broadcasting
        t = t[..., None, None, None]
        alpha_bar = alpha_bar[..., None, None, None]

        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * epsilon

        loss = self.loss(epsilon, model(xt, t))
        loss.backward()

        return loss.item()
    