import torch
from torch import nn 
from d2l import torch as d2l
import numpy as np
import random

def set_seed(seed = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_net(mode='modern'):
    if mode == 'modern':
        net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5,padding=2), nn.Sigmoid(),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Flatten(),
                            nn.Linear(400, 120), nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(120, 84), nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(84, 10))
    elif mode == 'old':
        net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5,padding=2), nn.Sigmoid(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Flatten(),
                            nn.Linear(400, 120), nn.Sigmoid(),
                            nn.Linear(120, 84), nn.Sigmoid(),
                            nn.Linear(84, 10))
    else:
        raise SyntaxError('请输入正确模式名:old/modern')
    return net

def test_net_shape(net):
    X = torch.rand((1,1,28,28),dtype=torch.float32)
    with open("model_check.log",'w') as f:
        for layer in net:
            X = layer(X)
            f.write(str(layer.__class__.__name__) + " output shape = " + str(X.shape) + '\n')

class Args:
    def __init__(self,batch_size = 256, lr = 0.9, num_epochs = 10):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def get_args(self):
        return self.batch_size, self.lr, self.num_epochs, self.device

def evaluate_accuracy(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
        
        metric = d2l.Accumulator(2)
        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(X, list):
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                metric.add(d2l.accuracy(net(X), y),y.numel())
    return metric[0] / metric[1]

def net_init(net,mode='xavior'):
    if mode=='xavior':
        def init_weight(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
        net.apply(init_weight)
    elif mode=='normal':
        def init_weight(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.normal_(m.weight)
        net.apply(init_weight)
    else:
        raise SyntaxError("不支持该模式")
    return net

def train(net, train_iter, test_iter, args):
    batch_size, lr, num_epochs, device = args.get_args()
    net.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=lr)
    # if mode == 'sgd':
    #     optimizer = torch.optim.SGD(net.parameters(),lr=lr)
    # elif mode == 'adam':
    #     optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    # else:
    #     raise SyntaxError("暂不支持该优化模式")
    
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],ylim=[0, 1.0],
                            legend=['train loss', 'train acc', 'test acc'])
    num_batches = len(train_iter)

    # 以下为训练代码
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            timer.stop()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


        
        
        

    





