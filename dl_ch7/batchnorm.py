import torch
from torch import nn 

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert(len(X.shape) in (2, 4)) # 判定X总是线性层/卷积层的input
        if(len(X.shape) == 2):
            mean = X.mean(dim = 0)
            var = ((X - mean) ** 2).mean(dim=0)
        elif(len(X.shape) == 4):
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self,num_dims,num_features): # num_features对特征数/通道数
        super(BatchNorm, self).__init__()
        assert(num_dims in (2, 4))
        if num_dims == 2:
            shape = (1, num_features)
        elif num_dims == 4:
            shape = (1, num_features, 1, 1)
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(X,self.gamma, self.beta, self.moving_mean, self.moving_var,0.00001, 0.9)
        return Y