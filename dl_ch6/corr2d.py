import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + 1 - h, X.shape[1] + 1 - w))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h,j:j + w] * K).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel_size, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))
        if bias == False:
            self.bias.requires_grad=False
            
        

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias
        
def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape) # 这里不是在填充，而是添加了批量数与通道数
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

if __name__ == "__main__":
    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1), stride=2)
    X = torch.rand((8, 8))
    print(comp_conv2d(conv2d, X).shape)

# conv2d = Conv2D((1,2), False)
# X = torch.ones((6, 8))
# X[:, 2:6] = 0
# Y = torch.zeros((6, 7))
# Y[:,1] = 1
# Y[:,5] = -1
# lr = 3e-2

# for i in range(10):
#     loss = (conv2d(X) - Y) ** 2
#     conv2d.zero_grad()
#     loss.sum().backward()
#     conv2d.weight.data[:] -= lr * conv2d.weight.grad
#     if (i + 1) % 2 == 0:
#         print(f'epoch {i+1}, loss {loss.sum():.3f}')
    
# print(conv2d.weight.data.reshape((1, 2)))







