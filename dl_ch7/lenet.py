import torch
from torch import nn 
from batchnorm import BatchNorm

def lenet():
    net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5,padding=2), BatchNorm(4, 6),nn.Sigmoid(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(6, 16, kernel_size=5),BatchNorm(4, 16), nn.Sigmoid(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Flatten(),
                            nn.LazyLinear(120),BatchNorm(2, 120), nn.Sigmoid(),nn.Dropout(0.5),
                            nn.LazyLinear(84), BatchNorm(2, 84), nn.Sigmoid(),nn.Dropout(0.5),
                            nn.LazyLinear(10))
    return net