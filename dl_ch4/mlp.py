import torch
from d2l import torch as d2l
from torch import nn  

def relu(X):
    zero = torch.zeros_like(X)
    return torch.max(X, zero)

def sigmoid(X):
    return 1 - 1 / (1 + torch.exp(X))

def L_infinite(X, W):
    torch.norm()


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return (H @ W2 + b2)

loss = nn.CrossEntropyLoss(reduction='none')



num_epochs = 30
lr = 0.1
batch_size = 256
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]

updater = torch.optim.SGD(params, lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
timer = d2l.Timer()
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
print(f'共训练{num_epochs}轮，每轮平均用时：{timer.stop() / num_epochs:.5f}sec')
d2l.predict_ch3(net, test_iter)
d2l.plt.show()