# 线性神经网络

## 线性回归
没什么特别的，主要是对损失


### 主要收获：
1. data_iter的实现思路（random.shuffle,yield）与简洁实现(data.Tensor.dataset, data.Dataloader)
2. 注意在合适的时候禁用梯度计算，在合适的时候让梯度重置
3. 每个epoch的训练循环：从train_iter中依次读取每个batch的features(X),labels(y)，计算l = loss(net(X),y),调用l.sum().backward()，用trainer更新params
4. 