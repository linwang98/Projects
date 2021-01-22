import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms,models
import matplotlib.pyplot as plt
import os
# 定义生成器网络G
class NetG(nn.Module):
    def __init__(self, ngf, nz):
        super(NetG, self).__init__()
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        # input opt.nz*1*1维的数据，opt.nz维的向量
        # output = (input - 1)*stride + output_padding - 2*padding + kernel_size
        # 把一个像素点扩充卷积，让机器自己学会去理解噪声的每个元素是什么意思。

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True)
        )
        # 输入一个4*4*ngf*8
        # layer2输出尺寸(ngf*4)x8x8
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        # 输入一个8*8*ngf*4
        # layer3输出尺寸(ngf*2)x16x16
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf*2 , 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True)
        )
        # 输入一个16*16*ngf*2
        # layer4输出尺寸(ngf)x32x32
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        # 输入一张32*32*ngf
        # layer5输出尺寸 3x96x96(1,80,80)
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
        # Tanh收敛速度快于sigmoid,远慢于relu,输出范围为[-1,1]，输出均值为0
            nn.Tanh()
        )

    # 定义NetG的前向传播
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out




# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()
        # layer1 输入 1 x 96 x 96, 输出 (ndf) x 32 x 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # input: (ndf, 32, 32)
        # layer2 输出 (ndf*2) x 16 x 16
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf,ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # input:(ndf *2, 16, 16)
        # layer3 输出 (ndf*4) x 8 x 8
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # input:(ndf *4, 8, 8)
        # layer4 输出 (ndf*8) x 4 x 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # input:(ndf *8, 4, 4)
        # output:(1, 1, 1)
        # layer5 输出一个数(概率)
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    # 定义NetD的前向传播
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # 展平后返回
        return out
