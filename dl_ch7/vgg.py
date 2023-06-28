import torch
from torch import nn 

def vgg_block(num_convs, in_channels, out_channels):
    layers = []

    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*layers)

def vgg(conv_arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 256))):
    conv_blks = []
    in_channels = 1
    for(num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    
    return nn.Sequential(*conv_blks,
                         nn.Flatten(),
                         nn.LazyLinear(4096),  nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.LazyLinear(4096),nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.LazyLinear(10)
                         )