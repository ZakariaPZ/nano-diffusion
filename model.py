import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        x_pooled = self.pool(x)
        return x, x_pooled

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_block(x)
        x = self.up_sample(x)
        return x

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
model = MiniUNet(in_channels=3, out_channels=1)

# Print the model architecture
print(model)
