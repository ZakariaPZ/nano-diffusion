import torch
import torch.nn as nn
import math

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
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()

        self.conv_block = nn.Sequential(
            DoubleConv(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        x_pooled = self.pool(x)
        return x, x_pooled

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = nn.Sequential(
            DoubleConv(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x_up, x_res):
        print('x_up', x_up.shape)   
        x_up = self.up_sample(x_up)
        print('x_up', x_up.shape)
        x = torch.cat((x_up, x_res), dim=1)
        x = self.conv_block(x)
        return x
    
# Our neural network architecture follows the backbone of PixelCNN++ [52], which is a U-Net [48]
# based on a Wide ResNet [72]. We replaced weight normalization [49] with group normalization [66]
# to make the implementation simpler. Our 32 × 32 models use four feature map resolutions (32 × 32
# to 4 × 4), and our 256 × 256 models use six. All models have two convolutional residual blocks
# per resolution level and self-attention blocks at the 16 × 16 resolution between the convolutional
# blocks [6]. Diffusion time t is specified by adding the Transformer sinusoidal position embedding [60]
# into each residual block.

# For now, the UNet is the basic version from https://arxiv.org/abs/1505.04597 with sinusoidal pos. embeddings
# TODO: add wide ResNet, self-attention, group normalization/weight normalization 
# TODO: add text conditioning 


class PositionalEmbedding(nn.Module):
    '''
    PE_(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE_(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    '''
    def __init__(self,
                 d_model,
                 timesteps=5000):
        super().__init__()

        self.pos_encodings = torch.zeros(timesteps, d_model)
        positions = torch.arange(timesteps).unsqueeze(-1)

        # Use log for numerical stability
        denom = torch.exp(math.log(10000) * (torch.arange(0, d_model, 2) / d_model)).unsqueeze(0) 

        self.pos_encodings[:, ::2] = torch.sin(positions/denom) # multiplication better?
        self.pos_encodings[:, 1::2] = torch.cos(positions/denom)

        self.pos_encodings.requires_grad = False

    def forward(self, x):
        return self.pos_encodings[:x.size()[1], :] # requires grad false? 
    
class MiniUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiniUNet, self).__init__()

        self.down_block1 = DownBlock(in_channels, 64)
        self.down_block2 = DownBlock(64, 128)
        self.middle = DownBlock(128, 256)  # Using DownBlock for the middle layer
        
        self.up_block1 = UpBlock(384, 128)
        self.up_block2 = UpBlock(192, 64)
        
        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Downsample
        x1, x1_pooled = self.down_block1(x)
        x2, x2_pooled = self.down_block2(x1_pooled)
        
        # Middle
        middle, _ = self.middle(x2_pooled)
        
        # Upsample
        x_up1 = self.up_block1(x2, middle)
        x_up2 = self.up_block2(x1, x_up1)
        
        # Output
        output = self.output_layer(x_up2)
        
        return output

# Example usage:
# Create a MiniUNet model with two downsampling and two upsampling blocks
# model = MiniUNet(in_channels=3, out_channels=1)

# Print the model architecture
# print(model)
