import torch
from d2l import torch as d2l

# x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# y = torch.relu(x)
# d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
# d2l.plt.show()
# y.backward(torch.ones_like(x), retain_graph=True)
# d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
# d2l.plt.show()

# x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# y = torch.sigmoid(x)
# d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
# d2l.plt.show()
# y.backward(torch.ones_like(x), retain_graph=True)
# d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
# d2l.plt.show()

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
d2l.plt.show()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
d2l.plt.show()