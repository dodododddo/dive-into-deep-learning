import torch
import random
from Timer import Timer

def synthetic_data(w,b,num_examples):
    X = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(X,w) + b
    y[:] = y + torch.normal(0,0.01,y.shape)
    return X, y.reshape(-1,1)

def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples - batch_size,batch_size):
        batch_indice = torch.tensor(indices[i : i + batch_size])
        yield features[batch_indice],labels[batch_indice]

def linreg(X,w,b):
    return torch.matmul(X,w) + b

def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
    

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)

'''看看样本散点图和每个batch情况'''
# print('features:', features[0],'\nlabel:', labels[0])
# d2l.set_figsize()
# d2l.plt.scatter(features[:, 0].detach().numpy(), labels.detach().numpy(), 1);
# d2l.plt.show()

# for X,y in data_iter(10,features,labels):
#     print(f'{X}\n{y}\n')

w = torch.normal(0,0.01,(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)
lr = 0.03
net = linreg
num_epochs = 3
batch_size = 10
loss = squared_loss
optimizate = sgd

timer = Timer()
for epoch in range(num_epochs):
    timer.start()
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y)
        l.sum().backward()
        optimizate([w,b],lr,batch_size)

    print(f'{timer.stop():.5f} sec\n')

    with torch.no_grad():
        train_loss = loss(net(features,w,b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')

print(f'batch_size = {batch_size}\nw = {w}\nb = {b}\n')