import torch
from torch import nn 
from d2l import torch as d2l

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, 1)

dropout1, dropout2 = 0.2, 0.3
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Dropout(dropout1), nn.Linear(512,256), nn.ReLU(), nn.Dropout(dropout2), nn.Linear(256, 10))
net.apply(init_weights)

batch_size = 256
lr = 0.1
num_epochs = 30

loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr, weight_decay=0.001)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
timer = d2l.Timer()
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
print(f'共训练{num_epochs}轮，每轮平均用时：{timer.stop() / num_epochs:.5f}sec')
d2l.predict_ch3(net, test_iter)
d2l.plt.show()
