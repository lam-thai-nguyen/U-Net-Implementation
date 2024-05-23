###########################
# Author: Lam Thai Nguyen #
###########################

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
           
        # Bottom layer 
        self.bottom_layer = DoubleConv(features[-1], features[-1]*2)
        
        # Decoder
        for feature in features[::-1]:
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, 2, 2)
            )
            self.decoder.append(DoubleConv(feature*2, feature))
        
        # Last layer
        self.conv1x1 = nn.Conv2d(features[0], out_channels, 1)
            
    def forward(self, x):
        skip_connections = []
        
        for conv in self.encoder:
            x = conv(x)
            skip_connections.insert(0, x)
            x = self.pool(x)
            
        x = self.bottom_layer(x)
        
        for i, conv in enumerate(self.decoder):
            x = conv(x)
            if i % 2 == 0:  # only when conv is ConvTranspose2d
                copy = skip_connections[i//2]
                
                if x.shape != copy.shape:
                    x = TF.resize(x, copy.shape[2:])
                
                paste = torch.cat([copy, x], dim=1)
                x = paste
        
        return self.conv1x1(x)
    

def test():
    print("Testing...")
    x = torch.randn((3, 1, 161, 161))
    model = UNET(1, 1)
    pred = model(x)
    assert pred.shape == x.shape
    print("Test passed.")
    

if __name__ == "__main__":
    test()
    