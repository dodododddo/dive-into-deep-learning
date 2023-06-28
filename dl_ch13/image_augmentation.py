import torch
import torchvision
from torch import nn 
from d2l import torch as d2l

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

d2l.set_figsize()
img = d2l.Image.open('girl.jpg')

augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), 
                                       torchvision.transforms.ColorJitter(0.5,0.5,0.5,0.5),
                                       torchvision.transforms.RandomResizedCrop(
                                       (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
                                       ])
apply(img, augs)
d2l.plt.show()
