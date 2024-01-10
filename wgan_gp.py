import os
import sys
import torch
import argparse
import torchvision

from torch.utils.data import DataLoader, dataset

from utils.lib_util import weights_init, gradient_penality, get_logger
from utils.model_util import Critic, Generator

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist',
                    help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake, default=mnist')
parser.add_argument('--batch_size', type=int, default=160,
                    help='input batch size, default=160')
parser.add_argument('--image_size', type=int, default=64,
                    help='the height / width of the input image to network, default=64')
parser.add_argument('--dim_noise', type=int, default=100,
                    help='size of the latent z vector, default=100')
parser.add_argument('--sgf', type=int, default=64,
                    help='the size of feature in generator, default=64')
parser.add_argument('--scf', type=int, default=64,
                    help='the size of feature in critic, default=64')
parser.add_argument('--epoch_gan', type=int, default=10,
                    help='number of epochs to train GAN, default=10')
parser.add_argument('--epoch_cri', type=int, default=5,
                    help='number of epochs to train critic, default=5')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate for Adam, default=0.0001')
parser.add_argument('--beta1', type=float, default=0,
                    help='beta1 for Adam, default=0')
parser.add_argument('--beta2', type=float, default=0.9,
                    help='beta2 for Adam, default=0.9')
parser.add_argument('--lambda_gp', type=float, default=10,
                    help='lambda_gp, default=10')
args = parser.parse_args()

log = get_logger()
if torch.cuda.is_available():
    device = torch.device('cuda')
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

img_dir = 'res/gan/img'
os.makedirs(img_dir, exist_ok=True)
gan_dir = 'res/gan'

# 下载数据集
if args.dataset == 'mnist':
    img_channel = 1
    size_img = [0.5 for _ in range(img_channel)]
    dataset = torchvision.datasets.MNIST(
        root='data',
        train=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((args.image_size, args.image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=size_img, std=size_img)]),
        download=True)
# dataset = torchvision.datasets.ImageFolder(
#     root=r"E:\conda_3\PyCharm\Transer_Learning\WGAN\WGANCode\data",
#     transform=transform)
# img_channel = 3

# 加载数据集
dataLoader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=4)

# 实例模型
net_generator = Generator(args.dim_noise, img_channel, args.sgf).to(device)
net_critic = Critic(img_channel, args.scf).to(device)
net_generator.apply(weights_init)
net_critic.apply(weights_init)
net_generator.train()
net_critic.train()

# 定义优化器
opt_gen = torch.optim.Adam(
    net_generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
opt_critic = torch.optim.Adam(
    net_critic.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# 定义随机噪声
fixed_noise = torch.randn(size=(25, args.dim_noise, 1, 1), device=device)
len_dataloader = len(dataLoader)
for epoch in range(args.epochs_gan):
    for batch_idx, (img, _) in enumerate(dataLoader):
        img = img.to(device)

        for _ in range(args.epoch_cri):
            noise = torch.randn(
                size=(img.shape[0], args.dim_noise, 1, 1), device=device)
            fake_img = net_generator(noise)
            critic_real = net_critic(img)
            critic_fake = net_critic(fake_img)

            gp = gradient_penality(net_critic, img, fake_img, device)
            loss_critic = torch.mean(critic_fake) - torch.mean(critic_real) \
                + args.lambda_gp * gp
            opt_critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
        # sys.exit()
        gen_fake = net_critic(fake_img)
        loss_gen = -torch.mean(gen_fake)
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 10 == 0 and batch_idx > 0:
            print(
                f"Epoch[{epoch}/{args.epoch_gan}] Batch {batch_idx}/{len_dataloader} Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}")

        if batch_idx % 100 == 0 and batch_idx > 0:
            net_generator.eval()
            net_critic.eval()
            fake_img = net_generator(fixed_noise)
            torchvision.utils.save_image(
                fake_img, f'{img_dir}/{epoch}-{batch_idx}.png', nrow=5, normalize=True)
            net_generator.train()
            net_critic.train()

torch.save(net_generator, f'{gan_dir}/generator.pth')
torch.save(net_critic, f'{gan_dir}/critic.pth')
