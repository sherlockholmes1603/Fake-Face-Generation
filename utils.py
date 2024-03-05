import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import imageio
import glob
import natsort
import matplotlib.pyplot as plt
import math
import itertools
import os


def get_data_loader(root, batch_size):
    
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = datasets.ImageFolder(root=root, transform=transform)

    # Data Loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader



def initialize_weight(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal(m.weight.data, 0.0, 0.02)


# Function to create a gif of image plots
def save_gif(path, fps, max_num=100, fixed_noise=False):
    if fixed_noise==True:
        path += 'fixed_noise/'
    else:
        path += 'variable_noise/'
    images = glob(path + '*.png')
    images = natsort.natsorted(images)
    images = images[:max_num]
    gif = []

    for image in images:
        gif.append(imageio.imread(image))
    imageio.mimsave(path+'animated.gif', gif, fps=fps)


# Function to generate image plots
def generate_images(epoch, path, fixed_noise, num_test_samples, netG, device, use_fixed=False):
    z = torch.randn((num_test_samples, 100, 1, 1)).to(device)
    size_figure_grid = int(math.sqrt(num_test_samples))
    title = None
  
    if use_fixed:
        generated_fake_images = netG(fixed_noise)
        path = os.path.join(path, "fixed_noise")
        title = 'Fixed Noise'
    else:
        generated_fake_images = netG(z)
        path = os.path.join(path, "variable_noise")
        title = 'Variable Noise'

    if not os.path.exists(path):
        os.makedirs(path)
  
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6,6))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
    for k in range(num_test_samples):
        i = k//size_figure_grid
        j = k%size_figure_grid
        ax[i,j].cla()
        ax[i,j].imshow((generated_fake_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2 )
    label = 'Epoch_{}'.format(epoch+1)
    fig.text(0.5, 0.04, label, ha='center')
    fig.suptitle(title)
    fig.savefig(path+label+'.png')





