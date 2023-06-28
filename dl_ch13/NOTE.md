# CV基础
## 图像增广(augmentation)
#### 目的
* 减小模型对特定属性的依赖，提高泛化能力
#### 常用手段
* 随机翻转
* 随机裁剪
* 随机调色
```
augs = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(), 
                    torchvision.transforms.ColorJitter(0.5,0.5,0.5,0.5),
                    torchvision.transforms.RandomResizedCrop(
                        (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
                    ])
```
#### 具体实现
1. 用compose分别定义对训练数据和测试数据的增广方式，并把ToTensor()包括在其中
```
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
```
2. 注意Compose总是接受列表作为参数
3. 将augs作为获取dataloader的函数的参数，来获取处理后的train_iter和test_iter


## 微调(fine-tuning)

