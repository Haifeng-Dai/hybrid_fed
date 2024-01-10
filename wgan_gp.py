import os
import sys
import time
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
parser.add_argument('--num_fake_img', type=int, default=50,
                    help='number of fake images to be generated, default=50')
args = parser.parse_args()

t = time.localtime()
log_path = f'./log/{t.tm_year}-{t.tm_mon}-{t.tm_mday}/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_path += f'{t.tm_hour}-{t.tm_min}-{t.tm_sec}.log'
log = get_logger(log_path)

if torch.cuda.is_available():
    device = 'cuda'
    log.info(f'device {device} is used.')
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        log.info('cudnn is actived.')
elif torch.backends.mps.is_available():
    device = 'mps'
    log.info(f'device {device} is used.')
else:
    device = 'cpu'
    log.info(f'device {device} is used.')

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

message = f"\n\
{'dataset':^13}:{args.dataset:^7}\n\
{'batch_size':^13}:{args.batch_size:^7}\n\
{'image_size':^13}:{args.image_size:^7}\n\
{'dim_noise':^13}:{args.dim_noise:^7}\n\
{'sgf':^13}:{args.sgf:^7}\n\
{'scf':^13}:{args.scf:^7}\n\
{'epoch_gan':^13}:{args.epoch_gan:^7}\n\
{'epoch_cri':^13}:{args.epoch_cri:^7}\n\
{'lr':^13}:{args.lr:^7}\n\
{'beta1':^13}:{args.beta1:^7}\n\
{'beta2':^13}:{args.beta2:^7}\n\
{'lambda_gp':^13}:{args.lambda_gp:^7}\n\
{'img_channel':^13}:{img_channel:^7}"
log.info(message)

# 加载数据集
dataLoader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=8)

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
for epoch in range(args.epoch_gan):
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

        message = f'Epoch[{epoch}/{args.epoch_gan}] Batch {batch_idx}/{len_dataloader} Loss D: {loss_critic:.2f}, Loss G: {loss_gen:.2f}'
        log.info(message)

        if batch_idx % 100 == 0 and batch_idx > 0:
            net_generator.eval()
            net_critic.eval()
            fake_img = net_generator(fixed_noise)
            torchvision.utils.save_image(
                fake_img, f'{img_dir}/{epoch}-{batch_idx}.png', nrow=5, normalize=True)
            message = 'fake images saved.'
            log.info(message)
            net_generator.train()
            net_critic.train()

torch.save(net_generator, f'{gan_dir}/generator.pth')
torch.save(net_critic, f'{gan_dir}/critic.pth')
message = 'model saved.'
log.info(message)
