import math
import torch

from torch import nn


def conv_cal(c, kernel_size, stride=None, padding=0, operation='conv'):
    '''
    卷积/池化操作后特征数量计算
    '''
    if stride == None:
        if operation == 'conv':
            stride = 1
        else:
            stride = kernel_size
    l_return = (c - kernel_size + 2 * padding) / stride + 1
    if operation == 'conv':
        return math.ceil(l_return)
    if operation == 'pool':
        return math.floor(l_return)


class LeNet5(nn.Module):
    '''
    修改后的LeNet5模型
    '''

    def __init__(self, h, w, c, num_classes):
        super(LeNet5, self).__init__()
        self.h = h
        self.w = w
        self.conv1 = nn.Sequential(
            nn.Conv2d(c, 6, 5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2))
        h_conv, w_conv = self.len_s()
        self.full_con = nn.Sequential(
            nn.Linear(16 * h_conv * w_conv, 120),
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
        h_conv = conv_cal(self.h, kernel_size=5)
        h_conv = conv_cal(h_conv, kernel_size=2, operation='pool')
        h_conv = conv_cal(h_conv, kernel_size=5)
        h_conv = conv_cal(h_conv, kernel_size=2, operation='pool')
        w_conv = conv_cal(self.w, kernel_size=5)
        w_conv = conv_cal(w_conv, kernel_size=2, operation='pool')
        w_conv = conv_cal(w_conv, kernel_size=5)
        w_conv = conv_cal(w_conv, kernel_size=2, operation='pool')
        return h_conv, w_conv


class MLP(nn.Module):

    def __init__(self, h, w, c, num_hidden=50, num_classes=10):
        super(MLP, self).__init__()

        self.input = nn.Sequential(
            nn.Linear(h * w * c, num_hidden),
            nn.ReLU(inplace=True))
        self.hidden = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(inplace=True),
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

    def __init__(self, h, w, c, num_classes):
        super(CNN, self).__init__()
        self.h = h
        self.w = w
        self.conv1 = nn.Sequential(
            nn.Conv2d(c, 6, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2))
        h_conv, w_conv = self.len_s()
        self.full_con = nn.Sequential(
            nn.Linear(16 * h_conv * w_conv, 64),
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
        h_conv = conv_cal(self.h, kernel_size=3)
        h_conv = conv_cal(h_conv, kernel_size=2, operation='pool')
        h_conv = conv_cal(h_conv, kernel_size=3)
        h_conv = conv_cal(h_conv, kernel_size=2, operation='pool')
        w_conv = conv_cal(self.w, kernel_size=3)
        w_conv = conv_cal(w_conv, kernel_size=2, operation='pool')
        w_conv = conv_cal(w_conv, kernel_size=3)
        w_conv = conv_cal(w_conv, kernel_size=2, operation='pool')
        return h_conv, w_conv


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False))

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(64 * 2,     64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
