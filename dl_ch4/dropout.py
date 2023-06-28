import torch
from torch import nn 
from d2l import torch as d2l

def dropout_layer(X, dropout):
    assert 0<=dropout<=1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1 - dropout)

class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True, dropout1=0.2, dropout2=0.5):
        super(MLP, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.linear1 = nn.Linear(num_inputs, num_hiddens1)
        self.linear2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.linear3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
        self.dropout1 = dropout1
        self.dropout2 = dropout2
    
    def testmod(self):
        self.training = False

    def trainmod(self):
        self.training = True

    def forward(self, X):
        H1 = self.relu(self.linear1(X.reshape((-1, self.num_inputs))))
        if self.training:
            H1 = dropout_layer(H1, self.dropout1)
        H2 = self.relu(self.linear2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout2)
        return self.linear3(H2)

nums_epochs = 30
lr = 0.1
batch_size = 256
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 512, 256
net = MLP(num_inputs, num_outputs, num_hiddens1, num_hiddens2, dropout2=0.3)
trainer = torch.optim.SGD(net.parameters(), lr, weight_decay=0.001)
loss = nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, nums_epochs, trainer)
net.testmod()
d2l.predict_ch3(net, test_iter)
d2l.plt.show()