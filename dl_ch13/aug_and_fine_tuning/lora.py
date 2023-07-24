import torch
from torch import nn 
from d2l import torch as d2l
import torchvision
import os
from train import train, Args, set_seed
import torch.nn.functional as F
from fine_tuning import Hotdog_loader

def lora_block(
        input_size:int,
        rank: int,
        output_size: int
    ):
    block = nn.Sequential(nn.Linear(input_size, rank), nn.Linear(rank, output_size))
    nn.init.xavier_uniform_(block[0].weight)
    nn.init.xavier_uniform_(block[1].weight)
    return block

class lora_fc(nn.Module):
    def __init__(self, fc, rank=4):
        super().__init__()
        self.fc = fc
        self.lora = lora_block(input_size=fc.in_features, rank=4, output_size=fc.out_features)
        self.output = nn.Linear(fc.out_features, 2)
        nn.init.xavier_uniform_(self.output.weight)
    
    def forward(self, X):
        return self.output(F.relu(self.fc(X) + self.lora(X)))

set_seed()
args = Args(256, 5e-5, 10)
loader = Hotdog_loader()
train_iter = loader.load(True, batch_size=args.batch_size)
test_iter = loader.load(False, batch_size=args.batch_size)

net = torchvision.models.resnet18(pretrained=True)
for param in net.parameters():
    param.requires_grad = False
output = lora_fc(net.fc)
net.fc = output

trainer = torch.optim.Adam(net.parameters(), lr=args.lr)
train(net, train_iter, test_iter, trainer, args)