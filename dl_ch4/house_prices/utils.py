import torch
from d2l import torch as d2l
import sys
from torch import nn 
import numpy as np
import pandas as pd
from download import *

DATA_HUB['kaggle_house_train'] = (  
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True, dropout1 = 0.2, dropout2 = 0.5, activate_func = nn.ReLU()):
        super(MLP, self).__init__()
        self.num_inputs, self.num_outputs = num_inputs, num_outputs
        self.is_training = is_training
        self.linear1 = nn.Linear(num_inputs, num_hiddens1)
        self.linear2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.linear3 = nn.Linear(num_hiddens2, num_outputs)
        nn.init.xavier_uniform_(self.linear1.weight, 1)
        nn.init.xavier_uniform_(self.linear2.weight, 1)
        nn.init.xavier_uniform_(self.linear3.weight, 1)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.activate_func = activate_func
        
    def forward(self, X):
        H1 = self.activate_func(self.linear1(X.reshape(-1, self.num_inputs)))
        if self.is_training:
            H1 = self.dropout1(H1)
        H2 = self.activate_func(self.linear2(H1))
        if self.is_training:
            H2 = self.dropout2(H2)
        return self.linear3(H2)
        
    def train_mod(self):
        self.is_training = True

    def test_mod(self):
        self.is_training = False
    

class logmseloss(nn.Module):
    def __init__(self):
        super(logmseloss, self).__init__()
        self.loss = nn.MSELoss()
    
    def forward(self, y_hat, y):
        clipped_preds = torch.clamp(y_hat, 1, float('inf'))
        return torch.sqrt(self.loss(torch.log(clipped_preds), torch.log(y)))

def get_net():
    num_inputs = 331
    num_hiddens1 = 256
    num_hiddens2 = 256
    num_outputs = 1

    # net = MLP(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(num_inputs, num_hiddens1), nn.ReLU(), nn.Dropout(0.2), 
                        nn.Linear(num_hiddens1, num_hiddens2), nn.ReLU(), nn.Dropout(0.5),
                        nn.Linear(num_hiddens2, num_outputs))
    # net = nn.Sequential(nn.Linear(num_inputs, num_outputs))
    def init_weight(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weight)
    return net


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    '''主循环'''
    
    loss = logmseloss()
    trainer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        with torch.no_grad():
            train_ls.append(loss(net(train_features), train_labels).item())
            if test_labels is not None:
                test_ls.append(loss(net(test_features), test_labels).item())
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx,:], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train == None:
            X_train, y_train = X_part, y_part
        else:
            X_train, y_train = torch.cat([X_train, X_part], 0), torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        # net = nn.Sequential(nn.Linear(331,1))
        # nn.init.normal_(net[0].weight)
        net = get_net().to(device)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        with open('k_fold.log', 'a') as file:
            sys.stdout = file
            print(f'折{i + 1},训练log rmse{float(train_ls[-1]):f}, '
                f'验证log rmse{float(valid_ls[-1]):f}',
                f'batch_size = {batch_size},lr={learning_rate},wd={weight_decay},num_epochs={num_epochs}')
    sys.stdout = sys.__stdout__
    return train_l_sum / k, valid_l_sum / k

def train_and_show(train_features, train_labels, test_features, num_epochs, lr, weight_decay, batch_size):
    net = get_net().to(device)
    train_ls, _ = train(net, train_features, train_labels, test_features, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    with open('train.log', 'a') as file:
        sys.stdout = file
        print(f'训练log rmse：{float(train_ls[-1])}',
            f'batch_size = {batch_size},lr={lr},wd={weight_decay},num_epochs={num_epochs}')
    sys.stdout = sys.__stdout__
    torch.save(net.state_dict(), 'model_params')

def pred_and_save(test_features, test_data):
    net = get_net().to(device)
    net.load_state_dict(torch.load('model_params'))
    preds = net(test_features).detach().cpu().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


