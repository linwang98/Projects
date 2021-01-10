import torch as t
import torch.nn as nn
from data import ChallengeDataset,get_train_dataset,get_validation_dataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from model import ResNet
import torchvision as tv


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# ＃从csv文件加载数据并执行train-test-split
# ＃这可以使用已经导入的pandas和sklearn.model_selection模块来完成
# TODO
#随机划分训练集和测试集

dataset =ChallengeDataset(flag=None,csv_file=None,transform= tv.transforms.Compose(tv.transforms.ToTensor()))
dataset.img_data = pd.read_csv('data.csv', sep=';')
# dataset = ChallengeDataset(flag=flag, csv_file=csv_file,ransform= tv.transforms.Compose(tv.transforms.ToTensor()))
dataset.train, dataset.val = train_test_split(dataset.img_data, random_state=42, test_size=0.2)
##准备好训练数据加载器
trainload =  t.utils.data.DataLoader(dataset=get_train_dataset(),batch_size=10)
testload = t.utils.data.DataLoader(dataset=get_validation_dataset(),batch_size=10)
#准备好模型
net = ResNet()
# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# ＃使用t.utils.data.DataLoader和ChallengeDataset对象分别为训练和验证集设置数据加载
# TODO

# create an instance of our ResNet model
# TODO

# ？？set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = one_hot_y)
# crit=t.nn.BCEWithLogitsLoss()
# crit=t.nn.MultiLabelSoftMarginLoss()
crit=t.nn.BCEWithLogitsLoss()
##优化方法
optimizer = t.optim.Adam(net.parameters(), lr=0.0001)

# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(model=net, crit=crit, optim=optimizer, train_dl=trainload, val_test_dl=testload,
                  cuda=True, early_stopping_patience=100000)
# ＃设置合适的损失准则（您可以在t.nn中找到预先实现的损失函数）
# ＃设置优化器（请参阅t.optim）
# ＃创建一个Trainer类型的对象并设置其早期停止条件
# TODO

# go, go, go... call fit on trainer
res = trainer.fit(200)
#TODO

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
plt.show()