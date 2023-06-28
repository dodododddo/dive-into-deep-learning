import torch
from torch import nn 
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, num_channels, stride=1, use_1x1conv=False):
        super(Residual, self).__init__()
        self.num_channels = num_channels
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        if use_1x1conv == True:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = self.conv2(F.relu(self.bn2(self.conv1(F.relu(self.bn1(X))))))
        if self.conv3:
            X = self.conv3(X)
        return Y + X
    
def resnet_block(num_channels, num_residuals=2, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(num_channels, stride=2, use_1x1conv=True))
        else:
            blk.append(Residual(num_channels))
    return blk

class bottleneck(nn.Module):
    def __init__(self, num_channels, middle_channels, use_1x1_conv=False):
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        self.conv1 = nn.LazyConv2d(middle_channels, kernel_size=1)
        self.conv2 = nn.LazyConv2d(middle_channels, kernel_size=3, padding=1)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1)
        if use_1x1_conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1)
        else:
            self.conv4 = None

    def forward(self, X):
        Y = self.conv1(F.relu(self.bn1(X)))
        Y = self.conv2(F.relu(self.bn2(Y)))
        Y = self.conv3(F.relu(self.bn3(Y)))
        if self.conv4:
            Y += self.conv4(X)
        return Y
          

def resnet():
    b1 = nn.Sequential(nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                   nn.LazyBatchNorm2d(), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, first_block=True))
    b3 = nn.Sequential(*resnet_block(128))
    b4 = nn.Sequential(*resnet_block(256))
    b5 = nn.Sequential(*resnet_block(512))
    b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.LazyLinear(10))
    net = nn.Sequential(b1, b2, b3, b4, b5, b6)
    return net


    

    
