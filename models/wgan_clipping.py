import torch
import torch.nn as nn
from utils import initialize_weight, get_data_loader
import torch.optim as optim
import os
import math
from matplotlib import pyplot as plt
import itertools





class Critic(nn.Module):
    def __init__(self, channel_img, feature_d):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channel_img, feature_d, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2),
            self._block(feature_d, feature_d*2, 4, 2, 1),
            self._block(feature_d*2, feature_d*4, 4, 2, 1),
            self._block(feature_d*4, feature_d*8, 4, 2, 1),
            nn.Conv2d(feature_d*8, 1, kernel_size = 4, stride = 2, padding = 0)
        )
        
    def _block(self, in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride, 
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.disc(x)
    


class Generator(nn.Module):
    def __init__(self, z_dim, channel_img, feature_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, feature_g*16, 4, 1, 0),
            self._block(feature_g*16, feature_g*8, 4, 2, 1),
            self._block(feature_g*8, feature_g*4, 4, 2, 1),
            self._block(feature_g*4, feature_g*2, 4, 2, 1),
            nn.ConvTranspose2d(feature_g*2, channel_img, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )

    def _block(self, in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channel,
                out_channel,
                kernel_size,
                stride, 
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.gen(x)
    


class WGAN_clipping(object):
    def __init__(self, epochs, cuda, batch_size, dataroot, lr=2e-4):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and cuda == "True") else "cpu")
        self.lr = lr
        self.batch_size = batch_size
        self.img_sze = 64
        self.channel_img = 3
        self.z_dim = 100
        self.num_epoch = epochs
        self.feature_disc = 64
        self.feature_gen = 64
        self.critic_iterations = 5
        self.weight_clip = 0.01
        self.dataroot = dataroot
        print(self.device)
        self.gan = Generator(self.z_dim, self.channel_img, self.feature_gen)
        self.critic = Critic(self.channel_img, self.feature_disc)
        initialize_weight(self.critic)
        initialize_weight(self.gan)

        

        self.opt_gan = optim.RMSprop(self.gan.parameters(), lr=lr)
        self.opt_critic = optim.RMSprop(self.critic.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

        self.fixed_noise = torch.randn(64, self.z_dim, 1, 1).to(self.device)

        self.critic.train()
        self.gan.train()

    def train(self, generate_images=False):
        self.loader = get_data_loader(self.dataroot, self.batch_size)
        for epoch in range(self.num_epoch):
            for i, (real, _) in enumerate(self.loader):
                real = real.to(self.device)

                
                for _ in range(self.critic_iterations):
                    noise = torch.randn((self.batch_size, self.z_dim, 1, 1)).to(self.device)
                    fake = self.gan(noise)
                    
                    critic_real = self.critic(real).reshape(-1)
                    critic_fake = self.critic(fake).reshape(-1)
                    loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
                    
                    self.critic.zero_grad()
                    loss_critic.backward(retain_graph=True)
                    self.opt_critic.step()
                    
                    for p in self.critic.parameters():
                        p.data.clamp_(-self.weight_clip, self.weight_clip)
                        
                    
                output = self.critic(fake).reshape(-1)
                loss_gen = -torch.mean(output)
                self.gan.zero_grad()
                loss_gen.backward()
                self.opt_gan.step()
                    
                    

                
                
                if (i+1) %100 == 0:
                    print("Epoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}".format(
            epoch+1, self.num_epoch, i+1, len(self.loader), loss_critic.item(), loss_gen.item()))
                
            if generate_images:
                self.gan.eval()
                generate_images(epoch, os.path.join(os.curdir(), "dcgan", "generated_images"), self.fixed_noise, 64, self.gan, self.device,  use_fixed=True)
                self.gan.train()
        
        self.save_model()


    def save_model(self):
        print("Saving the model")
        torch.save(self.gan.state_dict(), 'generator_WGAN_clipping.pth')
        torch.save(self.disc.state_dict(), 'discriminator_WGAN_clipping.pth')

    def load_weight(self, disc_path, gan_path):
        self.critic.load_state_dict(torch.load(disc_path))
        self.gan.load_state_dict(torch.load(gan_path))

    