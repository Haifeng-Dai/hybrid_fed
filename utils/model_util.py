import torch
import torchvision

from torch import nn


def conv_cal(c, kernel_size, stride=1, padding=0):
    '''
    卷积/核池化后特征数量计算公式
    '''
    l_return = (c - kernel_size + 2 * padding) / stride + 1
    if l_return % 1 == 0:
        return int(l_return)
    else:
        raise ValueError(
            'kernel size, stride or padding need to be adjusted.')


class LeNet5(nn.Module):
    '''
    修改后的LeNet5模型
    '''

    def __init__(self, c, w, h, num_classes):
        super(LeNet5, self).__init__()
        self.c = c
        self.w = w
        self.conv1 = nn.Sequential(
            nn.Conv2d(h, 6, 5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2))
        c_conv, w_conv = self.len_s()
        self.full_con = nn.Sequential(
            nn.Linear(16 * c_conv * w_conv, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True))
        self.output = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.full_con(x)
        x = self.output(x)
        return x

    # 计算卷积/池化层后的特征数
    def len_s(self):
        c_conv = conv_cal(self.c, 5)
        c_conv = conv_cal(c_conv, 2, 2)
        c_conv = conv_cal(c_conv, 5)
        c_conv = conv_cal(c_conv, 2, 2)
        w_conv = conv_cal(self.w, 5)
        w_conv = conv_cal(w_conv, 2, 2)
        w_conv = conv_cal(w_conv, 5)
        w_conv = conv_cal(w_conv, 2, 2)
        return c_conv, w_conv


class MLP(nn.Module):

    def __init__(self, c, w, h, num_hidden=200, num_classes=10):
        super(MLP, self).__init__()

        self.input = nn.Sequential(
            nn.Linear(c * w * h, num_hidden),
            nn.ReLU(inplace=True))
        self.hidden = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(inplace=True))
        self.output = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        return x


class CNN(nn.Module):

    def __init__(self, c, w, h, num_classes):
        super(CNN, self).__init__()
        self.c = c
        self.w = w
        self.conv1 = nn.Sequential(
            nn.Conv2d(h, 6, 3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2))
        c_conv, w_conv = self.len_s()
        self.full_con = nn.Sequential(
            nn.Linear(16 * c_conv * w_conv, 64),
            nn.ReLU(inplace=True))
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.full_con(x)
        x = self.output(x)
        return x

    # 计算卷积/池化层后的特征数
    def len_s(self):
        c_conv = conv_cal(self.c, 3)
        c_conv = conv_cal(c_conv, 2)
        c_conv = conv_cal(c_conv, 3)
        c_conv = conv_cal(c_conv, 2)
        w_conv = conv_cal(self.w, 3)
        w_conv = conv_cal(w_conv, 2)
        w_conv = conv_cal(w_conv, 3)
        c_conv = conv_cal(c_conv, 2)
        return c_conv, w_conv
