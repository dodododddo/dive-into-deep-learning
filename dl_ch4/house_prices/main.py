import torch
from utils import *


# train_data = pd.read_csv(download('kaggle_house_train'))
# test_data = pd.read_csv(download('kaggle_house_test'))

# all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:])) # 注意train_data中最后一列是labels
# numerial_features = all_features.dtypes[all_features.dtypes != 'object'].index
# all_features[numerial_features] = all_features[numerial_features].apply(lambda x: (x - x.mean()) / x.std())
# all_features[numerial_features] = all_features[numerial_features].fillna(0)
# all_features = pd.get_dummies(all_features, dummy_na=True)

# num_train = train_data.shape[0]
# train_features = torch.tensor(all_features[:num_train].values, dtype=torch.float32)
# test_features = torch.tensor(all_features[num_train:].values, dtype=torch.float32)
# train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
# torch.save([train_features, test_features, train_labels, test_data], 'processed_datas')

train_features, test_features, train_labels, test_data = torch.load('processed_datas')
train_features = train_features.to(device)
test_features = test_features.to(device)
train_labels = train_labels.to(device)


k = 5
batch_size = 64
lr = 0.01
wd = 0.00001
num_epochs = 2000


# train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, wd, batch_size)
timer = d2l.Timer()
train_and_show(train_features, train_labels, test_features, num_epochs, lr, wd, batch_size)
print(f'共训练{num_epochs}轮，平均每轮耗时{(timer.stop() / num_epochs):.5f}sec')
pred_and_save(test_features, test_data)
d2l.plt.show()