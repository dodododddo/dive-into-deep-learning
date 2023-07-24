import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from loader import load_data_bananas
from utils import *
from model import TinySSD
from train import train
import sys

set_seed(42)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
net:nn.Module

if len(sys.argv) == 1 or sys.argv[1] == 'predict':
    net = torch.load('TinySSD.pt')

elif len(sys.argv) >= 2 and sys.argv[1] == 'train':
    net = TinySSD(num_classes=1)
    test_net(net)
    args = Args(32, 5e-4, 40)
    train_iter, test_iter = load_data_bananas(args.batch_size)
    train(net, train_iter, test_iter, args, device)

else:
    raise SyntaxError("invalid mode!")  
 
# 加载图片并转为(批量，通道，h, w)的张量
X = torchvision.io.read_image('banana.jpeg').unsqueeze(0).float()
# 转为GBR图片？
img = X.squeeze(0).permute(1, 2, 0).long()
X = X.to(device)
result = predict(net, X).to(torch.device('cpu'))
display(img, result, threshold=0.9)
