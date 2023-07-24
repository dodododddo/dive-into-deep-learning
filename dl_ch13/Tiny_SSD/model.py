import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

def cls_predictor(num_inputs, num_anchors, num_classes):
    # 对特征图上每个像素的每个anchor预测给出num_classes + 1个预测值
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

def bbox_predictor(num_inputs, num_anchors):
    # 对特征图上每个像素的每个anchor预测给出offset(4维)
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

def flatten_pred(pred: torch.Tensor):
    # 真正的预测值仅在通道数维度中，则把通道数放到最后一维，展平为(批量数， 高 x 宽 x 通道数)，方便连接不同尺度的预测结果
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds: list[torch.Tensor]):
    return torch.cat([flatten_pred(pred) for pred in preds], dim=1)

def down_sample_blk(in_channels, out_channels):
    
    net = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.MaxPool2d(2) 
                        )
    return net

def base_net(num_filter=[3, 16, 32, 64]):
    blk = []
    for i in range(len(num_filter) - 1):
        blk.append(down_sample_blk(num_filter[i], num_filter[i + 1]))
    return nn.Sequential(*blk)

def get_blk(i: int):
    if i == 0:
        blk = base_net()
    
    elif i == 1:
        blk = down_sample_blk(64, 128)
    
    elif i == 4:
        blk = nn.AdaptiveAvgPool2d((1, 1))
  
    else:
        blk = down_sample_blk(128, 128)
    
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


class TinySSD(nn.Module):
    
    def __init__(self, 
                 num_classes,
                 sizes=[[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],[0.88, 0.961]], 
                 ratios=[[1, 2, 0.5]] * 5, 
                 ):
        super().__init__()
        self.num_classes = num_classes
        idx2inchannels = [64, 128, 128, 128, 128]
        self.sizes = sizes
        self.ratios = ratios
        self.num_anchors = len(sizes[0]) + len(ratios[0]) - 1
 
        self.blks = nn.ModuleList()
        self.cls_predictors = nn.ModuleList()
        self.bbox_predictors = nn.ModuleList()

        for i in range(5):
            self.blks.append(get_blk(i))
            self.cls_predictors.append(cls_predictor(idx2inchannels[i], self.num_anchors, num_classes))
            self.bbox_predictors.append(bbox_predictor(idx2inchannels[i], self.num_anchors))

    
    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(X, self.blks[i], self.sizes[i], self.ratios[i],self.cls_predictors[i], self.bbox_predictors[i])
        
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        
        return anchors, cls_preds, bbox_preds

class multiloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(reduction='none') # 不设置reduction='none'则后续无法reshape
        self.bbox_loss = nn.L1Loss(reduction='none')

    def forward(self, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        batch_size = cls_preds.shape[0]
        num_classes = cls_preds.shape[2]
        loss_cls = self.cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
        loss_bbox = self.bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
        return loss_cls + loss_bbox

