import torch
from torch import nn  
from d2l import torch as d2l

num_train, num_test, num_inputs, batch_size = 20, 10, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, num_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, num_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_panalty(w):
    return torch.sum(w.pow(2)) / 2

def l1_panalty(w):
    return torch.sum(w.abs())

def l_infinate_panalty(w):
    return torch.max(w.abs())


def train(lambd, train_iter, test_iter, panalty):
    w, b = init_params()
    num_epochs, lr = 100, 0.003
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * panalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                    d2l.evaluate_loss(net, test_iter, loss)))

    print('w的范数是：', torch.norm(w).item())


train(5, train_iter, test_iter, l1_panalty)
d2l.plt.show()