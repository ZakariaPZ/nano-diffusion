import torch
import torch.nn as nn
import math

# Our neural network architecture follows the backbone of PixelCNN++ [52], which is a U-Net [48]
# based on a Wide ResNet [72]. We replaced weight normalization [49] with group normalization [66]
# to make the implementation simpler. Our 32 × 32 models use four feature map resolutions (32 × 32
# to 4 × 4), and our 256 × 256 models use six. All models have two convolutional residual blocks
# per resolution level and self-attention blocks at the 16 × 16 resolution between the convolutional
# blocks [6]. Diffusion time t is specified by adding the Transformer sinusoidal position embedding [60]
# into each residual block.

# For now, the UNet is the basic version from https://arxiv.org/abs/1505.04597 with sinusoidal pos. embeddings
# TODO: add dropout (requires model.train), wide ResNet, self-attention, group normalization/weight normalization
# TODO: add text conditioning 


class DoubleConv(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(          
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 input_layer=False):
        super(DownBlock, self).__init__()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if not input_layer else nn.Identity()
        self.conv_block = DoubleConv(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv_block(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 t_dim=None):
        super(UpBlock, self).__init__()

        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = DoubleConv(in_channels, out_channels, kernel_size=3, padding=1)

        if t_dim:
            self.t_embedding = nn.Sequential(nn.Linear(t_dim, in_channels), nn.ReLU(inplace=True))

    def forward(self, x_up, x_res, t=None):
        if t is not None:
            x_up = x_up + self.t_embedding(t)[..., None, None]
        x_up = self.up_sample(x_up)
        x = torch.cat((x_up, x_res), dim=1)
        x = self.conv_block(x)
        return x
    
class SinusoidalTimeEmbedding(nn.Module):
    '''
    PE_(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE_(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    '''
    def __init__(self,
                 t_dim,
                 n_timesteps=1000):
        super().__init__()

        self.register_buffer('time_encodings', torch.zeros(n_timesteps, t_dim))
        timesteps = torch.arange(n_timesteps).unsqueeze(-1)

        # Use log for numerical stability
        denom = torch.exp(math.log(10000) * (torch.arange(0, t_dim, 2) / t_dim)).unsqueeze(0) 

        self.time_encodings[:, ::2] = torch.sin(timesteps/denom) # multiplication better?
        self.time_encodings[:, 1::2] = torch.cos(timesteps/denom)

        self.time_encodings.requires_grad = False

    def forward(self, t):

        return self.time_encodings[t, :] # requires grad false? 
    

class MiniUNet(nn.Module):
    def __init__(self,
                 t_dim = 64,
                 n_timesteps=1000):
        super(MiniUNet, self).__init__()

        self.t_embeddings = SinusoidalTimeEmbedding(t_dim=t_dim, n_timesteps=n_timesteps)

        # UNet architecture
        self.down_block1 = DownBlock(3, 64, input_layer=True)
        self.down_block2 = DownBlock(64, 128)

        self.middle = DownBlock(128, 256)
        
        self.up_block1 = UpBlock(256, 128, t_dim=t_dim)
        self.up_block2 = UpBlock(128, 64, t_dim=t_dim)
        
        self.output_layer = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x, t):
        t_embedding = self.t_embeddings(t)
        # Downsample
        down1 = self.down_block1(x)
        down2 = self.down_block2(down1)
        
        # Middle
        middle = self.middle(down2)

        # Upsample
        up1 = self.up_block1(middle, down2, t_embedding)
        up2 = self.up_block2(up1, down1, t_embedding)
        
        # Output
        output = self.output_layer(up2)
        
        return output

