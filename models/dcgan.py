import torch
import torch.nn as nn
from utils import initialize_weight, get_data_loader
import torch.optim as optim




class Discriminator(nn.Module):
    '''
    This is the discriminator for DCGAN
    '''
    def __init__(self, channel_img, feature_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channel_img, feature_d, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2),
            self._block(feature_d, feature_d*2, 4, 2, 1),
            self._block(feature_d*2, feature_d*4, 4, 2, 1),
            self._block(feature_d*4, feature_d*8, 4, 2, 1),
            nn.Conv2d(feature_d*8, 1, kernel_size = 4, stride = 2, padding = 0),
            nn.Sigmoid()
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
    

class DC_GAN(object):
    def __init__(self, epochs, cuda, batch_size, dataroot, lr=2e-4):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and cuda) else "cpu")
        self.lr = lr
        self.batch_size = batch_size
        self.img_sze = 64
        self.channel_img = 3
        self.z_dim = 100
        self.num_epoch = epochs
        self.feature_disc = 64
        self.feature_gen = 64
        self.print(self.device)
        self.gan = Generator(self.z_dim, self.channel_img, self.feature_gen)
        self.disc = Discriminator(self.channel_img, self.feature_disc)
        initialize_weight(self.disc)
        initialize_weight(self.gan)

        self.loader = get_data_loader(dataroot, batch_size)

        self.opt_gen = optim.Adam(self.gen.parameters(), lr=lr, betas = (0.5, 0.999))
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=lr, betas = (0.5, 0.999))
        self.criterion = nn.BCELoss()

        self.disc.train()
        self.gan.train()


    def train(self):
        for epoch in range(self.num_epoch):
            for i, (real, _) in enumerate(self.loader):
                real = real.to(self.device)
                noise = torch.randn((self.batch_size, self.z_dim, 1, 1)).to(self.device)
                fake = self.gen(noise)
                
                disc_real = self.disc(real).reshape(-1)
                loss_disc_real = self.criterion(disc_real, torch.ones_like(disc_real))
                
                disc_fake = self.disc(fake).reshape(-1)
                loss_disc_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
                
                loss_disc = (loss_disc_real + loss_disc_fake) / 2
                self.disc.zero_grad()
                loss_disc.backward(retain_graph=True)
                self.opt_disc.step()
                
                
                
                
                
                output = self.disc(fake).reshape(-1)
                loss_gen = self.criterion(output, torch.ones_like(output))
                self.gen.zero_grad()
                loss_gen.backward()
                self.opt_gen.step()
                
                
                if (i+1) %100 == 0:
                    print("Epoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}".format(
            epoch+1, self.num_epoch, i+1, len(self.loader), loss_disc.item(), loss_gen.item()))
                    
        self.save_model()
                    
    
    def save_model(self):
        print("Saving the model")
        torch.save(self.gan.state_dict(), 'generator_DCGAN.pth')
        torch.save(self.disc.state_dict(), 'discriminator_DCGAN.pth')
