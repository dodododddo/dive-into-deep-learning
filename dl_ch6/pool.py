import torch
from torch import nn 

def pool_2d(X, pool_size, mode='max'):
    pool_h, pool_w = pool_size
    Y = torch.zeros((X.shape[0] + 1 - pool_h, X.shape[1] + 1 - pool_w))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode=='max':
                Y[i, j] = torch.max(X[i : i + pool_h,j : j + pool_w])
            elif mode=='avg':
                Y[i, j] = torch.mean(X[i : i + pool_h,j : j + pool_w])
    return Y

# X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# print(pool_2d(X, (2, 2)))
# print(pool_2d(X, (2, 2), mode='avg'))

X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
X = torch.cat((X,X + 1) , 1)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
