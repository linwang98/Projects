import torch as t
import torch.nn as nn
import torch.nn.functional as F
# from torch import nn
# from torch.nn import functional as F
# import numpy as np
import torchvision as tv
# import operator

# data = tv.datasets.FashionMNIST(root="data", download=True)
# batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(len(data))), 100, False)
#each ResBlock consists (Conv2D, BatchNorm, ReLU) that is repeated twice.
class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels, stride):
        super(ResBlock, self).__init__()
        # self.conv = nn.Conv2d(out_channels=64,kernel_size7, stride=2)
        # self.pool = nn.MaxPool2d(pool_size=3, stride=2)
        # self.fc = nn.Linear(out_features)
        self.part1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
    )
        self.conv1 = nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1)
        self.batch1 = nn.BatchNorm2d(out_channels)

        # self.conv1 = nn.Conv2d(1, 100, 5)
        # self.conv2 = nn.Conv2d(100, 50, 5)
        # self.conv3 = nn.Conv2d(50, 5, 5)
        # self.fc1 = nn.Linear(5 * 256, 100)
        # self.fc2 = nn.Linear(100, 10)
        # self.sin = OurSin.apply # if in use, OurSin must be changed into float32
    def forward(self, x):
        # 1 × 1 convolution + batchnorm
        x0= self.conv1(x)
        x0= self.batch1(x0)
        x = self.part1(x)+x0

        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            #out channels, filter size, stride
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size = 7 ,  stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.layer1 = ResBlock(64, 64, 1)
        self.layer2 = ResBlock(64, 128, 2)
        self.layer3 = ResBlock(128, 256, 2)
        self.layer4 = ResBlock(256, 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.modules.flatten.Flatten()
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x