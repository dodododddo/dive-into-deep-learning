import torch 
from torch import nn 
from torch.nn import functional as F

def conv_block(num_channels):
    return nn.Sequential(nn.LazyBatchNorm2d(), nn.ReLU(), nn.LazyConv2d(num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for layer in self.net:
            Y = layer(X)
            X = torch.cat((Y, X), dim=1)
        return X
    
def transition_block(num_channels):
    return nn.Sequential(nn.LazyBatchNorm2d(),nn.ReLU(), nn.LazyConv2d(num_channels, kernel_size=1), nn.AvgPool2d(2, stride=2))

def densenet():
    b1 = nn.Sequential(nn.LazyConv2d(64, kernel_size=7, padding=3, stride=2), nn.LazyBatchNorm2d(), nn.ReLU(), nn.MaxPool2d(3, padding=1, stride=2))
    blks = []
    for i in range(4):
        blks.append(DenseBlock(4, 32))
        if i != 3:
            blks.append(transition_block(64))
    
    b3 = nn.Sequential(nn.LazyBatchNorm2d(), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.LazyLinear(10))

    net = nn.Sequential(b1, *blks, b3)
    return net
