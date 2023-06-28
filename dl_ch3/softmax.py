import torch
import random
from IPython import display
from d2l import torch as d2l
from load_faction_mnist import get_faction_mnist_labels

class Accumulator:
    def __init__(self,n):
        self.data = [0.0] * n
    
    def add(self,*args):
        self.data = [a + float(b) for a, b in zip(self.data,args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self,index):
        return self.data[index]


def softmax(X):
    C = -10
    X_exp = torch.exp(X + C)
    partition = X_exp.sum(axis=1,keepdim=True)
    return X_exp / partition

def net(X):
    return softmax(torch.matmul(X.reshape(-1,W.shape[0]),W) + b)

def cross_entropy(y_hat,y):
    return -1 * torch.log(y_hat[range(len(y_hat)),y])

def accuracy(y_hat,y):
    '''并未除去len(y)，保持可加性'''
    if(len(y_hat.shape) > 1 and y_hat.shape[1] > 1):
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    
    metric = Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())

    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        
        metric.add(float(l.sum()),accuracy(y_hat, y),y.numel())
    
    return metric[0] / metric[2],metric[1] / metric[2]

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metric = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metric + (test_acc,))
        
def predict(net, test_iter, n=6):
    assert n <= len(test_iter)
    rand = random.randint(1,len(test_iter) - 1)
    with torch.no_grad():
        for X, y in test_iter:
            if rand == 0:
                break
            rand -= 1

        trues, preds = get_faction_mnist_labels(y), get_faction_mnist_labels(net(X).argmax(axis=1))
        titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
        d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
        d2l.plt.show()


lr = 0.1
def updater(batch_size):
    return d2l.sgd([W,b], lr, batch_size)


if __name__ == "__main__":
    batch_size = 256
    num_inputs = 784
    num_outputs = 10
    num_epochs = 10
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    W = torch.normal(0,0.01,(num_inputs,num_outputs),requires_grad=True)
    b = torch.zeros(num_outputs,requires_grad=True)
    # y = torch.tensor([0,2])
    # y_hat = torch.tensor([[0.1,0.3,0.6],[0.2,0.3,0.5]])
    # print(cross_entropy(y_hat,y))
    # print(accuracy(y_hat,y))
    # print(evaluate_accuracy(net,test_iter))

    train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    predict(net, test_iter)