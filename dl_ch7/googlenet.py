import torch
from torch import nn 
from torch.nn import functional as F

class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.LazyConv2d(c1, kernel_size=1)
        self.p2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.p2_2 = nn.LazyConv2d(c2[1], kernel_size=3,padding=1)
        self.p3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.p3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3,padding=1, stride=1)
        self.p4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, X):
        p1 = F.relu(self.p1_1(X))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(X))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(X))))
        p4 = F.relu(self.p4_2(self.p4_1(X)))
        return torch.cat((p1, p2, p3, p4), dim=1)
    
def googlenet():
    b1 = nn.Sequential(nn.LazyConv2d(64, kernel_size=7, padding=3, stride=2),nn.ReLU(),nn.MaxPool2d(kernel_size=3, padding=1, stride=2))
    b2 = nn.Sequential(nn.LazyConv2d(64, kernel_size=1), nn.ReLU(), 
                       nn.LazyConv2d(192, kernel_size=3, padding=1),nn.ReLU(), 
                       nn.MaxPool2d(kernel_size=3, padding=1 ,stride=2))
    b3 = nn.Sequential(Inception(64, (96, 128), (16, 32), 32),
                       Inception(128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, padding=1, stride=2))
    b4 = nn.Sequential(Inception(192, (96, 208), (16, 48), 64),
                       Inception(160, (112, 224), (24, 64), 64),
                       Inception(128, (128, 256), (24, 64), 64),
                       Inception(112, (144, 288), (32, 64), 64),
                       Inception(256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, padding=1, stride=2))
    b5 = nn.Sequential(Inception(256, (160, 320), (32, 128), 128),
                   Inception(384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())
    b6 = nn.LazyLinear(10)
    net = nn.Sequential(b1, b2, b3, b4, b5, b6)
    return net

 