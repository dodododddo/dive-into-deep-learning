import torch
from torch import nn
import torchvision
import unittest
import itertools

def FCN(num_classes):
    pretrained_net = torchvision.models.resnet18(pretrained = True)
    net = nn.Sequential(*list(pretrained_net.children())[:-2])
    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module('trans_conv', nn.ConvTranspose2d(num_classes, num_classes,kernel_size=64, padding=16, stride=32))
    nn.init.xavier_uniform_(net.final_conv.weight)
    net.trans_conv.weight.data.copy_(bilinear_kernel(num_classes, num_classes, 64)) 

    return net

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    
    for i, j in itertools.product(range(in_channels), range(out_channels)):
        weight[i, j, :, :] = filt
    
    return weight


class TestBilinearKernel(unittest.TestCase):
    def test_odd_kernel_size(self):
        in_channels = 3
        out_channels = 4
        kernel_size = 5

        weight = bilinear_kernel(in_channels, out_channels, kernel_size)

        # Assert the shape of the weight tensor
        self.assertEqual(weight.shape, (in_channels, out_channels, kernel_size, kernel_size))

        # Assert the values of the weight tensor for the center pixel
        center_pixel_value = (1 - 0 / 3) * (1 - 0 / 3)
        self.assertEqual(weight[0, 0, 2, 2], center_pixel_value)

    def test_even_kernel_size(self):
        in_channels = 2
        out_channels = 2
        kernel_size = 4

        weight = bilinear_kernel(in_channels, out_channels, kernel_size)

        # Assert the shape of the weight tensor
        self.assertEqual(weight.shape, (in_channels, out_channels, kernel_size, kernel_size))

        # Assert the values of the weight tensor for the center pixels
        center_pixel_value = (1 - 0.5 / 2) * (1 - 0.5 / 2)
        self.assertEqual(weight[0, 0, 1, 1], center_pixel_value)
        self.assertEqual(weight[1, 1, 1, 1], center_pixel_value)

if __name__ == '__main__':
    unittest.main()
