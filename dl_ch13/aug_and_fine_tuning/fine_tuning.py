import torch
from torch import nn 
from d2l import torch as d2l
import torchvision
import os
from train import train, Args, set_seed
import torch.nn.functional as F

class Hotdog_loader(object):
    d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                                'fba480ffa8aa7e0febbb511d181409f899b9baa5')

    data_dir = d2l.download_extract('hotdog')

    normalize = torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize])

    test_augs = torchvision.transforms.Compose([
            torchvision.transforms.Resize([256, 256]),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize])
    
    def __init__(self):
        pass
        

    def load(self, is_train, batch_size):
        if is_train:
            train_img = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.train_augs)
            train_iter = torch.utils.data.DataLoader(train_img, batch_size=batch_size, shuffle=True)
            return train_iter
        else:
            test_img = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'test'), transform=self.test_augs)
            test_iter = torch.utils.data.DataLoader(test_img, batch_size=batch_size)
            return test_iter
        
if __name__ == '__main__':
    set_seed()
    args = Args(256, 5e-5, 10)
    loader = Hotdog_loader()
    train_iter = loader.load(True, batch_size=args.batch_size)
    test_iter = loader.load(False, batch_size=args.batch_size)

    finetuning_net = torchvision.models.resnet18(pretrained=True)
    finetuning_net.fc = nn.Linear(finetuning_net.fc.in_features, 2)
    nn.init.xavier_uniform_(finetuning_net.fc.weight)


params_hidden = [param for name, param in finetuning_net.named_parameters() if name not in ['fc.weight', 'fc.bias']]
trainer = torch.optim.Adam([{'params': params_hidden}, 
                            {'params': finetuning_net.fc.parameters(), 'lr': args.lr * 10}], 
                            lr=args.lr, weight_decay=0.001)

train(finetuning_net, train_iter, test_iter, trainer, args)


