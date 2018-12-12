'''
    based on: 
    https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    https://github.com/aelnouby/Text-to-Image-Synthesis/blob/master/utils.py
'''
import numpy as np
from torch import nn
from torch import  autograd
import torch
import os
import pdb

class Utils(object):
    @staticmethod
    def smooth_label(tensor, offset):
        return tensor + offset

    @staticmethod
    def save_checkpoint(netD, netG, dir_path, epoch):
        path = os.path.join(dir_path, '')
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(netD.state_dict(), '{0}/disc_{1}.pth'.format(path, epoch))
        torch.save(netG.state_dict(), '{0}/gen_{1}.pth'.format(path, epoch))

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)



