import torch
from torch import nn
from d2l import torch as d2l
from softmax import train, predict

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net.apply(init_weight)
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(),lr=0.1)

batch_size = 256
num_epochs = 10
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
train(net, train_iter, test_iter, loss, num_epochs, trainer)
predict(net, test_iter)
d2l.plt.show()

