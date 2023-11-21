import torch
from torch import nn
from .ddpm import DDPM

# No need for nn.Module...?
class DDIM(DDPM):

    def __init__(self, 
                 scheduler,
                 loss) -> None:
        super(DDIM, self).__init__(scheduler=scheduler, loss=loss)

    @torch.no_grad()
    def sample(self,
               model,
               device,
               n_samples,
               n_inference_steps,
               shape=(1, 28, 28)):
        '''
        DDIM deterministic sampling.
        '''
        timesteps = torch.linspace(0, self.scheduler.n_timesteps, n_inference_steps)
        model.eval()
        # Sample noise
        xt = torch.randn(n_samples, *shape).to(device)

        for i in torch.arange(n_inference_steps-1, 0, -1):
            i_batch = i.repeat(n_samples)
            timestep = timesteps[i_batch]
            alpha_bar_t = self.scheduler.alpha_bar[i_batch][..., None, None, None]
            alpha_bar_tm1 = self.scheduler.alpha_bar[i_batch - 1][..., None, None, None]

            pred_x0 = torch.sqrt(alpha_bar_tm1) * (xt - torch.sqrt(1 - alpha_bar_t) * model(xt, timestep.to(device))) / torch.sqrt(alpha_bar_t)
            correction = torch.sqrt(1 - alpha_bar_tm1) * model(xt, timestep.to(device)) 
            
            xtm1 = pred_x0 + correction
            xt = xtm1
        
        # xt *= torch.sqrt(self.scheduler.alpha_bar[0]) # Not sure about this?

        return xt.cpu().detach()
