import torch
from torch import nn
from d2l import torch as d2l
from utils import *
from model import multiloss
import sys

def train(net:nn.Module, train_iter, test_iter, args:Args, device):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, args.num_epochs],
                            legend=['train class error', 'train bbox mae', 'test class error', 'test bbox mae'])
    timer = d2l.Timer()
    batch_size, lr, num_epoches = args.get_args()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    loss = multiloss()
    net = net.to(device)

    for epoch in range(num_epoches):
        train_metric = d2l.Accumulator(4)
        test_metric = d2l.Accumulator(4)
        
        net.train()
        for features, targets in train_iter:
            timer.start()
            optimizer.zero_grad()
            features = features.to(device)
            targets = targets.to(device)

            anchors, cls_preds, bbox_preds = net(features)
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, targets)
            l = loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            optimizer.step()

            train_metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
        
        net.eval()
        for features, targets in test_iter:
            features = features.to(device)
            targets = targets.to(device)
            
            anchors, cls_preds, bbox_preds = net(features)
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, targets)
            test_metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())

        train_cls_error = 1 - train_metric[0] / train_metric[1]
        train_bbox_mae = train_metric[2] / train_metric[3]
        test_cls_error = 1 - test_metric[0] / test_metric[1]
        test_bbox_mae = test_metric[2] / test_metric[3]
        animator.add(epoch + 1, (train_cls_error, train_bbox_mae, test_cls_error, test_bbox_mae))
        torch.save(net, 'TinySSD.pt')

    with open("train_log.out",'a+') as f:
        temp = sys.stdout
        sys.stdout = f
        print('###########################################################')
        print(f'train_class_error {train_cls_error:.2e}, train_bbox_mae {train_bbox_mae:.2e}')
        print(f'test_class_error {test_cls_error:.2e}, test_bbox_mae {test_bbox_mae:.2e}')
        print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on 'f'{str(device)}\n')
        sys.stdout = temp
    
    d2l.plt.show()