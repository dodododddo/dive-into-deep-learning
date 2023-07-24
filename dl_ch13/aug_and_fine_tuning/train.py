import torch
import torchvision
from torch import nn 
from d2l import torch as d2l
import numpy as np
import random

def set_seed(seed = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

class Args:
    def __init__(self,batch_size = 128, lr = 0.01, num_epochs = 10):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def get_args(self):
        return self.batch_size, self.lr, self.num_epochs, self.device

def train(net, train_iter, test_iter, optimizer, args):
    batch_size, lr, num_epochs, device = args.get_args()
    net.to(device)
    loss = nn.CrossEntropyLoss()
    
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],ylim=[0, 1.0],
                            legend=['train loss', 'train acc', 'test acc'])
    num_batches = len(train_iter)

    # 以下为训练代码
    best_test_acc = 0.5
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
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
        if(test_acc > best_test_acc):
            save_model(net, 'model')
            best_test_acc = test_acc
        save_model(net, 'model', last_epoch=True)
        
    with open('train_datas.log','a+') as f:
        f.write(f'\nloss {train_l:.3f}, train acc {train_acc:.3f}, '
            f'test acc {test_acc:.3f}, ' f'best_test_acc {best_test_acc:.3f} \n')
        f.write(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
            f'on {str(device)}\n')
    d2l.plt.show()
        
def save_model(net, mode, last_epoch=False):
    if last_epoch:
        torch.save(net,mode + '.pt')
    else:
        torch.save(net, mode + '-best.pt')

def test_net_shape(net, device='cpu', image_size=224):
    X = torch.rand((1,1,image_size, image_size),dtype=torch.float32).to(device)
    with open("model_check.log",'w+') as f:
        for layer in net:
            X = layer(X)
            f.write(str(layer.__class__.__name__) + " output shape = " + str(X.shape) + '\n')



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
