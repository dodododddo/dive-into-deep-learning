import torch
from torch import nn 

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
              nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(),
              nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
              nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
    )

def nin():
    net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    nn.AdaptiveAvgPool2d((1,1)), # output_size:(1,10,1,1)
    nn.Flatten())

    return net
