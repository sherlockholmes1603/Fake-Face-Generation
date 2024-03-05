import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
import argparse
from models.dcgan import DC_GAN
from models.wgan_clipping import WGAN_clipping
from models.wgan_gp import WGAN_GP
import os



if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Pytorch implementation of GAN models.")
    

    args.add_argument('--model', type=str, default='DCGAN', choices=['DCGAN', 'WGAN-CP', 'WGAN-GP'])
    args.add_argument('--is_train', type=bool, default=False)
    args.add_argument('--dataroot', required=False, help='path to dataset')
    args.add_argument('--download', type=str, default='False')
    args.add_argument('--epochs', type=int, default=50, help='The number of epochs to run')
    args.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    args.add_argument('--cuda',  type=bool, default=True, help='Availability of cuda')

    args.add_argument('--load_D', type=str, default='False', help='Path for loading Discriminator network')
    args.add_argument('--load_G', type=str, default='False', help='Path for loading Generator network')

    args = args.parse_args()




    if args.model == "DCGAN":
        model = DC_GAN(args.epochs, args.cuda, args.batch_size, args.dataroot)
    elif args.model == "WGAN-CP":
        model = WGAN_clipping(args.epochs, args.cuda, args.batch_size, args.dataroot)
    elif args.model == "WGAN-GP":
        model = WGAN_GP(args.epochs, args.cuda, args.batch_size, args.dataroot)
    else:
        print("We don't have the model you want to train")
        exit()
    
    if args.is_train:
        if (args.load_G and args.load_D is not None):
            model.load_weight(args.load_D, args.load_G)
        print("Training")
        model.train()
    else:
        model.load_weight(args.load_D, args.load_G)

    model.gan.eval()

    
    generate_images(0, os.path.join(os.curdir, "test", args.model), None, 64, model.gan, model.device)

