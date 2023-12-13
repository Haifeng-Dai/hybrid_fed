import torch
from torch import nn


class LeNet5(nn.Module):

    def __init__(self, l, w, h, num_classes):
        super(LeNet5, self).__init__()

        self.l = l
        self.w = w
        self.conv1 = nn.Sequential(
            nn.Conv2d(h, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        l_conv, w_conv = self.len_s()
        self.full_con = nn.Sequential(
            nn.Linear(16 * l_conv * w_conv, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Linear(84, num_classes)

    # 卷积/核池化后特征数量计算公式
    def conv_cal(self, l, kernel_size, stride, padding=0):
        l_return = (l - kernel_size + 2 * padding) / stride + 1
        # return int(l_return)
        if l_return % 1 == 0:
            return int(l_return)
        else:
            raise ValueError('kernel size, stride or padding need to be adjusted.')

    # 计算卷积/池化层后的特征数
    def len_s(self):
        l_conv = self.conv_cal(self.l, 5, 1)
        l_conv = self.conv_cal(l_conv, 2, 2)
        l_conv = self.conv_cal(l_conv, 5, 1)
        l_conv = self.conv_cal(l_conv, 2, 2)
        w_conv = self.conv_cal(self.w, 5, 1)
        w_conv = self.conv_cal(w_conv, 2, 2)
        w_conv = self.conv_cal(w_conv, 5, 1)
        w_conv = self.conv_cal(w_conv, 2, 2)
        return l_conv, w_conv


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.full_con(x)
        x = self.output(x)
        return x



class MLP(nn.Module):

    def __init__(self, num_input, num_hidden, num_output):
        super(MLP, self).__init__()

        self.input = nn.Sequential(
            nn.Linear(num_input, num_hidden),
            nn.ReLU(inplace=True)
        )
        self.hidden = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        return x



class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.fc(x)
        return x