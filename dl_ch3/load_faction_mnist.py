import torchvision
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms
from Timer import Timer
d2l.use_svg_display()

def get_faction_mnist_labels(labels):
    test_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [test_labels[int(i)] for i in labels]

'''简单看看图片是什么样'''
# X,y = next(iter(data.DataLoader(mnist_train,batch_size=18))))
# d2l.show_images(X.reshape(18,28,28),2,9,titles=get_faction_mnist_labels(y))
# d2l.plt.show()

def load_data_faction_mnist(batch_size, resize=None, num_download_workers=4):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=num_download_workers),
           data.DataLoader(mnist_test, batch_size, shuffle=False,num_workers=num_download_workers))


def get_load_time(batch_size):
    timer = Timer()
    timer.start()
    train_iter, test_iter = load_data_faction_mnist(batch_size, resize=None)
    return timer.stop()

if __name__ == "__main__":
    batch_size_list = list(range(2,512,1))
    d2l.plot(batch_size_list,[[get_load_time(batch_size) for batch_size in batch_size_list]],'batch_size','load_time')
    d2l.plt.show()