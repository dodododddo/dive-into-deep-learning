from d2l import torch as d2l
from utils import *


set_seed(42)
image_size = 96
mode = 'resnet-18'
net = get_net(mode)
args = Args(batch_size = 256, lr=0.005, num_epochs=10)
test_net_shape(net, image_size=image_size)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=args.batch_size, resize=image_size)
train(net, train_iter, test_iter, args, mode=mode)
d2l.plt.show()