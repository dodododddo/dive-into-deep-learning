import torch
import torchvision
from d2l import torch as d2l

def load_cifar_10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root='../data', train=is_train, transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader

def augs(mode='train'):
    if mode == 'train':
        augs = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor()])
    elif mode == 'test':
        augs = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        # 仍应保留列表形式来确保可迭代性
    else:
        raise SyntaxError()
    return augs





