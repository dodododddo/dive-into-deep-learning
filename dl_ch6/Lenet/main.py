import torch
from torch import nn 
from d2l import torch as d2l
from utils import *

set_seed(42)
net = get_net()
# net = get_net('old')
# test_net_shape(net)
net_init(net)
args = Args(lr=0.9, num_epochs=100)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=args.batch_size)
train(net, train_iter, test_iter, args)
d2l.plt.show()
