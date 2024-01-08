import os
import torch
from torch.utils.data import DataLoader

import torch.nn as nn

from torchvision import datasets, transforms
from torchvision.utils import save_image

from tqdm import tqdm


class Discriminator(nn.Module):  # 定义判别器(WS-divergence)
    def __init__(self, img_shape=(1, 28, 28)):  # 初始化方法
        super(Discriminator, self).__init__()  # 继承初始化方法
        self.img_shape = img_shape  # 图片形状

        self.linear1 = nn.Linear(
            self.img_shape[0] * self.img_shape[1] * self.img_shape[2], 512)  # linear映射
        self.linear2 = nn.Linear(512, 256)  # linear映射
        self.linear3 = nn.Linear(256, 1)  # linear映射
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)  # leakyrelu激活函数

    def forward(self, x):  # 前传函数
        x = torch.flatten(x, 1)  # 输入图片从三维压缩至一维特征向量，(n,1,28,28)-->(n,784)
        x = self.linear1(x)  # linear映射，(n,784)-->(n,512)
        x = self.leakyrelu(x)  # leakyrelu激活函数
        x = self.linear2(x)  # linear映射,(n,512)-->(n,256)
        x = self.leakyrelu(x)  # leakyrelu激活函数
        x = self.linear3(x)  # linear映射,(n,256)-->(n,1)

        return x  # 返回近似拟合的Wasserstein距离


class Generator(nn.Module):  # 定义生成器
    def __init__(self, img_shape=(1, 28, 28), latent_dim=100):  # 初始化方法
        super(Generator, self).__init__()
        self.img_shape = img_shape  # 图片形状
        self.latent_dim = latent_dim  # 噪声z的长度

        self.linear1 = nn.Linear(self.latent_dim, 128)  # linear映射
        self.linear2 = nn.Linear(128, 256)  # linear映射
        self.bn2 = nn.BatchNorm1d(256, 0.8)  # bn操作
        self.linear3 = nn.Linear(256, 512)  # linear映射
        self.bn3 = nn.BatchNorm1d(512, 0.8)  # bn操作
        self.linear4 = nn.Linear(512, 1024)  # linear映射
        self.bn4 = nn.BatchNorm1d(1024, 0.8)  # bn操作
        self.linear5 = nn.Linear(
            1024, self.img_shape[0] * self.img_shape[1] * self.img_shape[2])  # linear映射
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)  # leakyrelu激活函数
        self.tanh = nn.Tanh()  # tanh激活函数，将输出压缩至（-1.1）

    def forward(self, z):  # 前传函数
        z = self.linear1(z)  # linear映射,(n,100)-->(n,128)
        z = self.leakyrelu(z)  # leakyrelu激活函数
        z = self.linear2(z)  # linear映射,(n,128)-->(n,256)
        z = self.bn2(z)  # 一维bn操作
        z = self.leakyrelu(z)  # leakyrelu激活函数
        z = self.linear3(z)  # linear映射,(n,256)-->(n,512)
        z = self.bn3(z)  # 一维bn操作
        z = self.leakyrelu(z)  # leakyrelu激活函数
        z = self.linear4(z)  # linear映射,(n,512)-->(n,1024)
        z = self.bn4(z)  # 一维bn操作
        z = self.leakyrelu(z)  # leakyrelu激活函数
        z = self.linear5(z)  # linear映射,(n,1024)-->(n,784)
        z = self.tanh(z)  # tanh激活函数
        # 从一维特征向量扩展至三维图片，(n,784)-->(n,1,28,28)
        z = z.view(-1, self.img_shape[0], self.img_shape[1], self.img_shape[2])

        return z  # 返回生成的图片


def cal_gp(D, real_imgs, fake_imgs, cuda):  # 定义函数，计算梯度惩罚项gp
    # 真假样本的采样比例r，batch size个随机数，服从区间[0,1)的均匀分布
    r = torch.rand(size=(real_imgs.shape[0], 1, 1, 1))
    if cuda:  # 如果使用cuda
        r = r.cuda()  # r加载到GPU
    # 输入样本x，由真假样本按照比例产生，需要计算梯度
    x = (r * real_imgs + (1 - r) * fake_imgs).requires_grad_(True)
    d = D(x)  # 判别网络D对输入样本x的判别结果D(x)
    fake = torch.ones_like(d)  # 定义与d形状相同的张量，代表梯度计算时每一个元素的权重
    if cuda:  # 如果使用cuda
        fake = fake.cuda()  # fake加载到GPU
    g = torch.autograd.grad(  # 进行梯度计算
        outputs=d,  # 计算梯度的函数d，即D(x)
        inputs=x,  # 计算梯度的变量x
        grad_outputs=fake,  # 梯度计算权重
        create_graph=True,  # 创建计算图
        retain_graph=True  # 保留计算图
    )[0]  # 返回元组的第一个元素为梯度计算结果
    gp = ((g.norm(2, dim=1) - 1) ** 2).mean()  # (||grad(D(x))||2-1)^2 的均值
    return gp  # 返回梯度惩罚项gp


if __name__ == "__main__":
    # 训练参数
    total_epochs = 100  # 训练轮次
    batch_size = 64  # 批大小
    lr_D = 4e-3  # 判别网络D学习率
    lr_G = 1e-3  # 生成网络G学习率
    num_workers = 8  # 数据加载线程数
    latent_dim = 100  # 噪声z长度
    image_size = 28  # 图片尺寸
    channel = 1  # 图片通道
    a = 10  # 梯度惩罚项系数
    clip_value = 0.01  # 判别器参数限定范围
    dataset_dir = "./data/"  # 训练数据集路径
    gen_images_dir = "./tmp/"  # 生成样例图片路径
    cuda = True if torch.cuda.is_available() else False  # 设置是否使用cuda
    os.makedirs(dataset_dir, exist_ok=True)  # 创建训练数据集路径
    os.makedirs(gen_images_dir, exist_ok=True)  # 创建样例图片路径
    image_shape = (channel, image_size, image_size)  # 图片形状

    # 模型
    D = Discriminator(image_shape)  # 实例化判别器
    G = Generator(image_shape, latent_dim)  # 实例化生成器
    if cuda:  # 如果使用cuda
        D = D.cuda()  # 模型加载到GPU
        G = G.cuda()  # 模型加载到GPU

    # 数据集
    transform = transforms.Compose(  # 数据预处理方法
        [transforms.Resize(image_size),  # resize
         transforms.ToTensor(),  # 转为tensor
         transforms.Normalize([0.5], [0.5])]  # 标准化
    )
    dataloader = DataLoader(  # dataloader
        dataset=datasets.MNIST(  # 数据集选取MNIST手写体数据集
            root=dataset_dir,  # 数据集存放路径
            train=True,  # 使用训练集
            download=True,  # 自动下载
            transform=transform  # 应用数据预处理方法
        ),
        batch_size=batch_size,  # 设置batch size
        num_workers=num_workers,  # 设置读取数据线程数
        shuffle=True  # 设置打乱数据
    )

    # 优化器
    optimizer_D = torch.optim.Adam(
        D.parameters(), lr=lr_D)  # 定义判别网络Adam优化器，传入学习率lr_D
    optimizer_G = torch.optim.Adam(
        G.parameters(), lr=lr_G)  # 定义生成网络Adam优化器，传入学习率lr_G

    # 训练循环
    for epoch in range(total_epochs):  # 循环epoch
        pbar = tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{total_epochs}', postfix=dict,
                    mininterval=0.3)  # 设置当前epoch显示进度
        LD = 0
        LG = 0
        for i, (real_imgs, _) in enumerate(dataloader):  # 循环iter
            print(real_imgs.shape)
            break
        break
        #     if cuda:  # 如果使用cuda
        #         real_imgs = real_imgs.cuda()  # 数据加载到GPU
        #     bs = real_imgs.shape[0]  # batchsize

        #     # 开始训练判别网络D
        #     optimizer_D.zero_grad()  # 判别网络D清零梯度
        #     z = torch.randn((bs, latent_dim))  # 生成输入噪声z，服从标准正态分布，长度为latent_dim
        #     if cuda:  # 如果使用cuda
        #         z = z.cuda()  # 噪声z加载到GPU
        #     fake_imgs = G(z).detach()  # 噪声z输入生成网络G，得到生成图片，并阻止其反向梯度传播
        #     gp = cal_gp(D, real_imgs, fake_imgs, cuda)
        #     # 判别网络D的损失函数，相较于WGAN，增加了梯度惩罚项a*gp
        #     loss_D = -torch.mean(D(real_imgs)) + \
        #         torch.mean(D(fake_imgs)) + a * gp
        #     loss_D.backward()  # 反向传播，计算当前梯度
        #     optimizer_D.step()  # 根据梯度，更新网络参数
        #     LD += loss_D.item()  # 累计判别网络D的loss

        #     # 开始训练生成网络G
        #     optimizer_G.zero_grad()  # 生成网络G清零梯度
        #     gen_imgs = G(z)  # 噪声z输入生成网络G，得到生成图片
        #     loss_G = -torch.mean(D(gen_imgs))  # 生成网络G的损失函数
        #     loss_G.backward()  # 反向传播，计算当前梯度
        #     optimizer_G.step()  # 根据梯度，更新网络参数
        #     LG += loss_G.item()  # 累计生成网络G的loss

        #     # 显示判别网络D和生成网络G的损失
        #     pbar.set_postfix(
        #         **{'D_loss': loss_D.item(), 'G_loss': loss_G.item()})
        #     pbar.update(1)  # 步进长度
        # pbar.close()  # 关闭当前epoch显示进度
        # print("total_D_loss:%.4f,total_G_loss:%.4f" % (
        #     LD / len(dataloader), LG / len(dataloader)))  # 显示当前epoch训练完成后，判别网络D和生成网络G的总损失
        # save_image(gen_imgs.data[:25], "%s/ep%d.png" % (gen_images_dir, (epoch + 1)), nrow=5,
        #            normalize=True)  # 保存生成图片样例(5x5)
