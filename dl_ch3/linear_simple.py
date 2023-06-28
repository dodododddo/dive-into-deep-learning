from d2l import torch as d2l
import torch
from torch.utils import data
from Timer import Timer
from torch import nn

def load_array(data_array,batch_size,is_train=True):
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size = 10
num_epochs = 3
true_w, true_b = torch.tensor([2,-3.4]), 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
data_iter = load_array((features,labels),batch_size)

net = nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(),lr=0.03)
timer = Timer()

for epoch in range(num_epochs):
    timer.start()
    for X,y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    print(f'{timer.stop():.5f}sec\n')
    l = loss(net(features),labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

