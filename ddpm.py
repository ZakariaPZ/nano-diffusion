import torch


class DDPM():

    def __init__(self, 
                 T,
                 dim) -> None:

        self.T = T 
        self.timesteps = torch.arange(0, T)
        self.dim = dim
        self.beta = torch.linspace(0, 1, T) # linear schedule
        self.alpha_bar = torch.cumprod(1 - self.betas, dim=0)


    def forward_process(self,
                        x0,
                        t):
        '''
        DDPM forward process. Take image x0 and noise to timestep t.
        '''

        eps = torch.randn((self.dim, self.dim))
        alpha_bar = self.alpha_bar[t]

        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * eps
        
        return xt


    def sample(self,
               model):
        '''
        DDPM reverse process. Predict noise in 
        '''
        # Sample noise
        xt = torch.randn((self.dim, self.dim))

        for t in self.timesteps[::-1]:
            alpha_t = 1 - self.beta[t]
            alpha_bar = self.alpha_bar[t]

            if t > 1:
                z = torch.randn((self.dim, self.dim))
            else:
                z = torch.zeros((self.dim, self.dim))

            sigma_t = torch.sqrt(self.beta[t])
            xtm1 = 1/torch.sqrt(alpha_t) * (xt - (1-alpha_t)/torch.sqrt(1-alpha_bar) * model(xt)) + sigma_t * z
            xt = xtm1
        
        return xt 
            
        