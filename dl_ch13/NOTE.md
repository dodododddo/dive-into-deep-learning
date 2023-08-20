# CV基础

## 1. 图像增广(augmentation)

#### 1.1 动机

* 减小模型对特定属性的依赖，提高泛化能力

#### 1.2 常用手段

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

###### 1.2.1 具体实现

1. 用compose分别定义对训练数据和测试数据的增广方式，并把ToTensor()包括在其中

```
def augs(mode='train'):
    if mode == 'train':
        augs = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor()])
    elif mode == 'test':
        # 仍应保留列表形式来确保可迭代性
        augs = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    else:
        raise SyntaxError()
    return augs
```

2. 注意Compose总是接受列表作为参数
3. 将augs作为获取dataloader的函数的参数，来获取处理后的train_iter和test_iter
<br><br>
## 2. 微调(fine-tuning)

#### 2.1 动机

网络在大数据集上学到的特征提取能力对完成目标数据集上的特定任务有帮助。与其选择为特定任务重新训练模型，可以用预训练的、在较大数据集上训练的模型进行微调来完成任务。

#### 2.2 常用手段

* 全参数：直接把输出层删去，改为目标任务的输出层，冻结输出层前的参数(或用较小的学习率更新)，重新学习新输出层的参数
* LoRA：在每个全连接层的权重矩阵外加一条低秩矩阵的旁路，冻结原权重矩阵，学习低秩矩阵参数
  训练后只需将低秩矩阵加到原权重上即可，不影响推理速度
  ![LoRA](https://pic3.zhimg.com/80/v2-e972b451c1a21c95e0b1a5483227a5f2_720w.webp)
<br>
## 3. 任务:目标检测
检测图像中目标的所在的位置范围与及其类别，以下是常用的评价指标
* 分类性能: mAP，其中AP(Average Precision)指对某一类别预测的PR曲线(Precision-Recall)的面积, mAP为多个类别的AP平均值
* 定位性能: IoU，检测框与真实框的交并比
* 其他性能: fps等


模型分类: one-stage/two-stage, anchor-based/anchor-free
经典模型: SSD(单发多框检测)、Fast-RCNN、YOLO等

#### 3.1 概念: 锚框
anchor-based的目标检测任务的单个标注样本为一张图片上的一个锚框，包括以下信息:
* 锚框的中央像素相对当前图片尺寸的坐标
* 锚框的宽与高(也是相对当前图片尺寸的值)
* 锚框的类别(包括背景)
* 锚框与被分配的真实框之间的编码距离

相应的，对图片上的每个锚框我们需要预测以下信息:
* 锚框属于每个类别(包括背景)的概率
* 锚框与被分配的真实框之间的编码距离：$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right)$
其中常数默认值为$\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x=\sigma_y=0.1, \sigma_w=\sigma_h=0.2$
你可以调控这些参数来使模型更关注边界框大小/中心位置/不同轴向的差异

不管是训练样本还是测试样本，锚框都需依据以下原则产生:
* 根据size(锚框边长占原图边长的比例)、ratio(锚框宽高比)以每一像素作为中心生成锚框，一般只考虑第一个size与所有ratio的组合以及第一个ratio与所有size的组合
* 对于size较大的锚框，我们期望用于预测较大尺度的目标，故不必逐个像素生成，而可以对像素进行采样。常用的方法是在尺寸较小的特征图上逐像素生成锚框，这等价于在原图上隔几个像素生成一个锚框。由于锚框中记录的各类参数均为相对值，故特征图上的锚框可以直接与真实框计算编码距离。要注意的是在特征图上取锚框仅仅是一种处理上的技巧，并不意味着我们试图检测特征图上的某个目标，我们依旧考虑的是原图的目标，只是参考的是更高抽象级别的特征

在处理标注样本时，每个锚框都会被分配一个真实框或背景，以便后续计算编码距离，分发的原则如下:
* 与所有真实框IoU都小于阈值的锚框会被标注为背景，在后续计算中会被掩码处理
* 与自身IoU值最高的真实框将被分配给该锚框，并标注为该真实框的类型(one-hot)，计算编码距离

最后，我们用非极大值抑制预测输出，即我们只考虑那些置信度(对应各类别的最高概率)大于阈值的锚框，并解码我们预测的编码距离来画出我们预测的物体位置

**需要注意的是，我们需要优化的损失部分不是锚框到真实框的编码距离，锚框只是一个介质，要优化的是类别预测与真实类别之间的交叉熵，以及预测的到真实框编码距离与标注的真实框编码距离之间的差距(通常用smooth-L1)**
<br><br>
#### 3.2 经典模型：SSD(单发多框检测)
###### 3.2.1 SSD的基本原理
backbone: VGG-16、resnet-18等确保能让特征图大小逐步减小的网络

每一层网络要做的事情: 
* backbone从当前特征图中提取特征并将新的特征图送入下一层网络
* 该层的类别预测器与距离预测器分别预测在当前特征图上生成的锚框的类别预测与编码距离预测，这些预测以同尺度特征图的形式呈现，每个像素对应以该像素为中心生成的不同锚框的预测值，每个通道则对应某个锚框的某个类别或某个锚框的编码距离的某个分量

训练: 分别融合不同层的类型预测值与距离预测值，分别计算损失，最后融合两种损失
推理: 分别融合不同层的类型预测值与距离预测值，解码并给出预测的物体位置及其类别
<br>
###### 3.2.2 SSD的实现细节
* cls_predictor输出的通道数和bbox_predictor输出的通道数
```
def cls_predictor(num_inputs, num_anchors, num_classes):
    # 对特征图上每个像素的每个anchor预测给出num_classes + 1个预测值
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

def bbox_predictor(num_inputs, num_anchors):
    # 对特征图上每个像素的每个anchor预测给出offset(4维)
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

```
* 用permute交换维度，用flatten按某一维度展平张量，从而使原来不同型的张量可以被拼接并保留一定程度上的结构
```
def flatten_pred(pred: torch.Tensor):
    # 真正的预测值仅在通道数维度中，则把通道数放到最后一维
    # 展平为(批量数， 高 x 宽 x 通道数)，方便连接不同尺度的预测结果
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds: list[torch.Tensor]):
    return torch.cat([flatten_pred(pred) for pred in preds], dim=1)
```
* 模型的数据流动比较复杂，需要用到多个同类旁支输出时，不能直接用list(优化器识别不到list中的参数)，而应用nn.ModuleList
```
self.blks = nn.ModuleList()
self.cls_predictors = nn.ModuleList()
self.bbox_predictors = nn.ModuleList()
```
* 注意保持每个anchor与其对应预测的匹配，最后解码与输出环节还需要用到anchor

* 考虑到类别预测时有大量负类存在，我们希望模型更关注困难分类的情形(即p的最大值较小),我们可以该用焦点损失$- \alpha (1-p_j)^{\gamma} \log p_j$



<br><br>
## 4. 任务:语义分割
目标: 获取图像中每个像素归属的语义类别，不对不同的实例进行区分
```
input: 3通道图像
output: num_classes个通道的伪图像
final_output: 最终判别每个像素的类型，并映射为一种颜色形成图像
``````
<br>
#### 4.1 概念：转置卷积
input的每个像素值依次与Kernel做矩阵数乘，按stride逐步移动相加
![trans_conv(stride=1)](https://zh-v2.d2l.ai/_images/trans_conv.svg)
![trans_conv(stride=2)](https://zh-v2.d2l.ai/_images/trans_conv_stride2.svg)
* stride影响中间结果叠加的方式
* padding用于对最终结果进行裁减

做kernel2matrix的展开后，可以发现转置卷积对应的矩阵形状恰与同参数的卷积对应矩阵形状呈现转置关系，故可以以此为标准来推算得到目标大小图像需要的转置矩阵参数
<br>
#### 4.2 概念: 采样
###### 4.2.1 下采样
* 无需学习：取对角线均值，池化
* 需学习：stride > 1的卷积层
###### 4.2.2 上采样
* 无需学习：双线性插值
* 需学习：stride > 1的转置卷积层
* 可以用双线性插值来初始化转置卷积层
<br>
#### 4.3 经典模型：全卷积网络(FCN)
![FCN](https://zh-v2.d2l.ai/_images/fcn.svg)
* resnet提取特征，生成特征图
* 1x1卷积层把通道数映射为num_classes
* 转置卷积层将特征图映射到与原图像等大的伪图像，每一像素的通道维存储该像素属于每一类别的概率
<br>
#### 4.4 经典模型：Unet
关键点：Encoder-Decoder架构 + shortcut(在通道方向concat实现)
![unet](https://pic1.zhimg.com/70/v2-314b7c7cea86d556540c380d7b2944d1_1440w.avis?source=172ae18b&biz_tag=Post)

* 对比FCN上采样过程的一步到位，Unet用双线性插值 + conv的方式逐步上采样，并通过shortcut不断融合该层次的特征，做到关注不同层次的特征
* loss选用加权的交叉熵函数，权重为$(1 - \frac{N_c}{N})$