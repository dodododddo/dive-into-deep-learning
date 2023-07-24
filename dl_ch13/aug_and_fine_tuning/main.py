from net import get_net
from train import *
from image_process import *

set_seed()

all_images = torchvision.datasets.CIFAR10(train=True, root="../../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

net = get_net(mode='resnet')
# test_net_shape(net)
args = Args(256, 0.001, 200)

train_augs = augs('train')
test_augs = augs('test')

train_iter = load_cifar_10(True, train_augs, args.batch_size)
test_iter = load_cifar_10(False, test_augs, args.batch_size)
trainer = torch.optim.Adam(net.parameters(),lr=args.lr)

train(net, train_iter, test_iter, trainer, args)
d2l.plt.show()