import torch
from torch import nn 

def alexnet():
    net = nn.Sequential(
              nn.LazyConv2d(96, kernel_size=11, padding=1, stride=4), nn.ReLU(),
              nn.MaxPool2d(kernel_size=3, stride=2),
              nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
              nn.MaxPool2d(kernel_size=3, stride=2),
              nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
              nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
              nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
              nn.MaxPool2d(kernel_size=3, stride=2),
              nn.Flatten(),
              nn.LazyLinear(4096), nn.ReLU(),
              nn.Dropout(0.5),
              nn.LazyLinear(4096), nn.ReLU(),
              nn.Dropout(0.5),
              nn.LazyLinear(10)
        )