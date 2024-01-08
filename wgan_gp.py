import os
import torch
import torchvision
from utils.data_util import *
from utils.lib_util import *
from utils.model_util import *
total_epochs = 100
batch_size = 160
lr_D = 4e-3  # 判别网络D学习率
lr_G = 1e-3  # 生成网络G学习率
num_workers = 8  # 数据加载线程数
latent_dim = 100  # 噪声z长度
image_size = 28  # 图片尺寸
channel = 1  # 图片通道
a = 10  # 梯度惩罚项系数
clip_value = 0.01  # 判别器参数限定范围
dataset_dir = "./data/"  # 训练数据集路径
gen_images_dir = "./gan_img/"  # 生成样例图片路径
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
transform = torchvision.transforms.Compose(  # 数据预处理方法
    [torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])])
dataloader = DataLoader(  # dataloader
    dataset=torchvision.datasets.MNIST(
        root='./data/',
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
        if cuda:  # 如果使用cuda
            real_imgs = real_imgs.cuda()  # 数据加载到GPU
        bs = real_imgs.shape[0]  # batchsize

        # 开始训练判别网络D
        optimizer_D.zero_grad()  # 判别网络D清零梯度
        z = torch.randn((bs, latent_dim))  # 生成输入噪声z，服从标准正态分布，长度为latent_dim
        if cuda:  # 如果使用cuda
            z = z.cuda()  # 噪声z加载到GPU
        fake_imgs = G(z).detach()  # 噪声z输入生成网络G，得到生成图片，并阻止其反向梯度传播
        gp = cal_gp(D, real_imgs, fake_imgs, cuda)
        # 判别网络D的损失函数，相较于WGAN，增加了梯度惩罚项a*gp
        loss_D = -torch.mean(D(real_imgs)) + \
            torch.mean(D(fake_imgs)) + a * gp
        loss_D.backward()  # 反向传播，计算当前梯度
        optimizer_D.step()  # 根据梯度，更新网络参数
        LD += loss_D.item()  # 累计判别网络D的loss

        # 开始训练生成网络G
        optimizer_G.zero_grad()  # 生成网络G清零梯度
        gen_imgs = G(z)  # 噪声z输入生成网络G，得到生成图片
        loss_G = -torch.mean(D(gen_imgs))  # 生成网络G的损失函数
        loss_G.backward()  # 反向传播，计算当前梯度
        optimizer_G.step()  # 根据梯度，更新网络参数
        LG += loss_G.item()  # 累计生成网络G的loss

        # 显示判别网络D和生成网络G的损失
        pbar.set_postfix(
            **{'D_loss': loss_D.item(), 'G_loss': loss_G.item()})
        pbar.update(1)  # 步进长度
    pbar.close()  # 关闭当前epoch显示进度
    print("total_D_loss:%.4f,total_G_loss:%.4f" % (
        LD / len(dataloader), LG / len(dataloader)))  # 显示当前epoch训练完成后，判别网络D和生成网络G的总损失
    save_image(gen_imgs.data[:25], "%s/ep%d.png" % (gen_images_dir, (epoch + 1)), nrow=5,
               normalize=True)  # 保存生成图片样例(5x5)
