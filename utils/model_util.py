import math
import torch
import torch.nn as nn


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


class Generator(torch.nn.Module):
    def __init__(self, channels_noise, img_channel, num_feature):
        super(Generator, self).__init__()
        # self.conv1 =

        self.gen = torch.nn.Sequential(
            # imgsize: 4 x 4
            self._block(in_channels=channels_noise,
                        out_channels=num_feature * 16,
                        kernel_size=(4, 4),
                        stride=(1, 1),
                        padding=0),
            # imgsize: 8 x 8
            self._block(in_channels=num_feature * 16,
                        out_channels=num_feature * 8,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1),
            # imgsize: 16 x 16
            self._block(in_channels=num_feature * 8,
                        out_channels=num_feature * 4,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1),
            # imgsize: 32 x 32
            self._block(in_channels=num_feature * 4,
                        out_channels=num_feature * 2,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1),
            # imgsize: N x 3 x 64 x 64
            torch.nn.ConvTranspose2d(
                in_channels=num_feature * 2,
                out_channels=img_channel,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1)),
            torch.nn.Tanh())

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU())
        return self.conv

    def forward(self, input):
        x = self.gen(input)
        return x


class Critic(torch.nn.Module):
    def __init__(self, img_channel, num_features):
        super(Critic, self).__init__()
        self.disc = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=img_channel, out_channels=num_features, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            self._block(in_channels=num_features, out_channels=num_features * 2, kernel_size=(4, 4), stride=(2, 2),
                        padding=(1, 1)),
            self._block(in_channels=num_features * 2, out_channels=num_features * 4, kernel_size=(4, 4), stride=(2, 2),
                        padding=(1, 1)),
            self._block(in_channels=num_features * 4, out_channels=num_features *
                        8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            torch.nn.Conv2d(in_channels=num_features*8, out_channels=1, kernel_size=(4, 4), stride=(2, 2), padding=0))

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            torch.nn.InstanceNorm2d(out_channels, affine=True),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))
        return self.conv

    def forward(self, input):
        x = self.disc(input)
        return x


if __name__ == '__main__':
    G = Generator(1, 100, 64)
    # G = Generator1(1)
    D = Discriminator(1, 64)

    noise = torch.randn(2, 100, 1, 1)
    # print(noise.shape)
    a = G(noise)
    # print(a.shape)
    # b = D(a)
    # print(b.shape)
    # print(b)
    # c = G(torch.randn(2, 100, 1, 1))
    # gp_a = compute_gradient_penalty(D, a, c)
    # print(gp_a)
