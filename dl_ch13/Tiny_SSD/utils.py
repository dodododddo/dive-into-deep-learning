import torch
from torch.nn import functional as F
from torch import nn
from d2l import torch as d2l
import numpy as np
import random
import sys

def set_seed(seed = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

class Args:
    def __init__(self,batch_size = 128, lr = 0.01, num_epochs = 10):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

    def get_args(self):
        return self.batch_size, self.lr, self.num_epochs

def save_model(net, mode, last_epoch=False):
    if last_epoch:
        torch.save(net,mode + '.pt')
    else:
        torch.save(net, mode + '-best.pt')

def test_net(net, device='cpu', image_size=256):
    X = torch.rand((32,3,image_size, image_size),dtype=torch.float32).to(device)
    anchors, cls_preds, bbox_preds = net(X)

    with open("model_shape.out",'w+') as f:
        temp = sys.stdout
        sys.stdout = f
        print('output anchors:', anchors.shape)
        print('output class preds:', cls_preds.shape)
        print('output bbox preds:', bbox_preds.shape)
    sys.stdout = temp

def cls_eval(cls_preds, cls_labels):

    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

def predict(net:nn.Module, X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X)
    # 先对每个类别评分做softmax，再把通道(对应锚框)放到最后一维
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    d2l.plt.show()