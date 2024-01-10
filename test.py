import torch
import os
import torchvision

from torchvision.utils import save_image

from utils.lib_util import *

torch.set_printoptions(
    precision=2,
    threshold=1000,
    edgeitems=5,
    linewidth=1000,
    sci_mode=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

os.makedirs("images", exist_ok=True)

n_epochs = 200
batch_size = 160
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu = 8
latent_dim = 100
img_size = 28
channels = 1
n_critic = 5
sample_interval = 400
lambda_gp = 10

# Initialize generator and discriminator
Gen = Generator(1, 100, 64).to(device)
Dis = Discriminator(1, 64).to(device)

# Configure data loader
os.makedirs("./data/", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./data/",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(
    Gen.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(
    Dis.parameters(), lr=lr, betas=(b1, b2))

# ----------
#  Training
# ----------
batches_done = 0
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        imgs = imgs.to(device)
        optimizer_D.zero_grad()

        # Generate a batch of images
        z = torch.randn(imgs.shape[0], latent_dim, 1, 1).to(device)
        fake_imgs = Gen(z)

        # Real images
        real_validity = Dis(imgs)
        # Fake images
        fake_validity = Dis(fake_imgs)
        # Gradient penalty
        gp = cal_gp(Dis, imgs.data, fake_imgs.data, device)
        # Adversarial loss
        d_loss = torch.mean(fake_validity) - \
            torch.mean(real_validity) + lambda_gp * gp

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------
            # Generate a batch of images
            fake_imgs = Gen(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = Dis(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            if batches_done % sample_interval == 0:
                save_image(fake_imgs.data[:25], "images/%d.png" %
                           batches_done, nrow=5, normalize=True)

            batches_done += n_critic
