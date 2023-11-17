import torch
from torch import nn

# No need for nn.Module...?
class DDPM(nn.Module):

    def __init__(self, 
                 scheduler,
                 loss) -> None:
        super(DDPM, self).__init__()

        self.scheduler = scheduler
        self.loss = loss

    @torch.no_grad()
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

        for t in torch.arange(self.scheduler.n_timesteps-1, -1, -1):
            t_batch = t.repeat(n_samples)
            alpha_t = 1 - self.scheduler.beta[t_batch][..., None, None, None]
            alpha_bar = self.scheduler.alpha_bar[t_batch][..., None, None, None]
            sigma_t = torch.sqrt(self.scheduler.beta[t_batch])[..., None, None, None]

            # Change to one line
            if t > 0:
                z = torch.randn(n_samples, *shape).to(device)
            else:
                z = torch.zeros(n_samples, *shape).to(device)

            xtm1 = 1/torch.sqrt(alpha_t) * (xt - (1-alpha_t)/torch.sqrt(1-alpha_bar) * model(xt, t_batch)) + sigma_t * z
            xt = xtm1
        
        return xt.cpu().detach()
    
    def train(self,
              x0,
              model,
              batch_size,
              shape):
        '''
        Training algorithm for DDPM. Algorithm 1 in the paper.
        '''
        model.train()

        t = torch.randint(0, self.scheduler.n_timesteps, (batch_size,))
        alpha_bar = self.scheduler.alpha_bar[t]
        z = torch.randn(batch_size, *shape).to(x0.device)

        # Unsqueeze to match dimensions of epsilon for broadcasting
        alpha_bar = alpha_bar[..., None, None, None]

        ## Use scheduler forward process
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * z

        loss = self.loss(z, model(xt, t))
        loss.backward()

        return loss.item()
    