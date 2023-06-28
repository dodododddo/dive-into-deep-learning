import torch
from d2l import  torch as d2l
from torch import nn 
from torch.nn import functional as F 

class MLP(nn.Module):
    def __init__(self, num_hiddens, num_outputs):
        super(MLP, self).__init__()
        self.hidden = nn.LazyLinear(num_hiddens)
        self.out = nn.LazyLinear(num_outputs)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

class Mysequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module
    
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
    
    def __getitem__(self, i):
        assert 0 <= i < len(self._modules)
        return self._modules[i]
            

class RobustNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weight = torch.randn((num_inputs, num_outputs), requires_grad=True)
    
    def forward(self, X):
        out = torch.zeros((X.shape[0], self.num_outputs))
        for j in range(self.num_outputs):
            for i in range(X.shape[0]):
                out[i, j] = torch.max(torch.abs(X[i,:] - self.weight[:, j]))
        return out



    

shared = RobustNet(300, 300)
net = Mysequential(shared, nn.ReLU(), shared, nn.LazyLinear(1))
X = torch.randn((5, 300))
print(X)
l = net(X)
print(l)
