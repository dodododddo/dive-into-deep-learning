import torch
from torch import nn 
from d2l import torch as d2l
from corr2d import corr2d

def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for (x, k) in zip(X, K))

def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K])

def corr2d_multi_in_out_1x1(X, K):
    ci, h, w = X.shape
    co = K.shape[0]
    X = X.reshape((ci, h * w))
    K = K.reshape((co, ci))
    return (K @ X).reshape((co,h,w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6



    # X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
    #             [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    # K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    # K = torch.stack((K, K + 1, K + 2), 0)
    # print(K.shape)

    # print(corr2d_multi_in_out(X, K))
