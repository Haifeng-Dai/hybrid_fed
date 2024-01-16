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


def critic_block(in_channels, out_channels, kernel_size, stride, padding):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False),
        torch.nn.InstanceNorm2d(out_channels, affine=True),
        torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))
    return block


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


class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=image_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1)
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(intermediate_channels * 4))
        layers.append(
            block(self.in_channels, intermediate_channels,
                  identity_downsample, stride))
        self.in_channels = intermediate_channels * 4
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]}


def vgg_layer(in_channels, architecture):
    layers = []
    for x in architecture:
        if type(x) == int:
            out_channels = x
            layers += [nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)),
                nn.BatchNorm2d(x),
                nn.ReLU()]
            in_channels = x
        elif x == "M":
            layers += [nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=(2, 2))]
    return nn.Sequential(*layers)


class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, VGG_type='VGG16'):
        super(VGG_net, self).__init__()
        self.conv_layers = vgg_layer(in_channels, VGG_type)
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x


def generator_block(in_channels, out_channels, kernel_size, stride, padding):
    block = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False),
        torch.nn.BatchNorm2d(num_features=out_channels),
        torch.nn.ReLU())
    return block


class Generator(torch.nn.Module):
    def __init__(self, channels_noise, img_channel, num_feature):
        super(Generator, self).__init__()
        # imgsize: 4 x 4
        self.conv1 = generator_block(
            in_channels=channels_noise,
            out_channels=num_feature * 16,
            kernel_size=(4, 4),
            stride=(1, 1),
            padding=0)
        # imgsize: 8 x 8
        self.conv2 = generator_block(
            in_channels=num_feature * 16,
            out_channels=num_feature * 8,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=1)
        # imgsize: 16 x 16
        self.conv3 = generator_block(
            in_channels=num_feature * 8,
            out_channels=num_feature * 4,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=1)
        # imgsize: 32 x 32
        self.conv4 = generator_block(
            in_channels=num_feature * 4,
            out_channels=num_feature * 2,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=1)
        # imgsize: N x 3 x 64 x 64
        self.conv5 = torch.nn.ConvTranspose2d(
            in_channels=num_feature * 2,
            out_channels=img_channel,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1))
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.tanh(x)
        return x


class Critic(torch.nn.Module):
    def __init__(self, img_channel, num_features):
        super(Critic, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=img_channel, out_channels=num_features, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = critic_block(
            in_channels=num_features, out_channels=num_features * 2,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1))
        self.conv3 = critic_block(
            in_channels=num_features * 2, out_channels=num_features * 4,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1))
        self.conv4 = critic_block(
            in_channels=num_features * 4, out_channels=num_features *
            8, kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1))
        self.conv5 = torch.nn.Conv2d(
            in_channels=num_features*8,
            out_channels=1,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
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
