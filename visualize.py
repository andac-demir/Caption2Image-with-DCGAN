'''
    based on:
    https://github.com/aelnouby/Text-to-Image-Synthesis/blob/master/utils.py

    Notes on running visdom:
    First, find the IP address of your remote GPU by typing ifconfig
    on the SSH screen as a command.
    Start the visdom server via python -m visdom.server 
    before running main.py. 
    Then on your browser go to <remote IP address>:8097
    to see the Visdom plots and figures that your visdom server is running. 
'''
from visdom import Visdom
import numpy as np
import torchvision
from PIL import ImageDraw, Image, ImageFont
import torch
import pdb

class VisdomPlotter(object):
    def __init__(self):
        self.viz = Visdom(port=8097, server="http://localhost")
        self.plots = {}

    def plot(self, var_name, split_name, x, y, xlabel='epoch'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), 
                                                 Y=np.array([y,y]), 
                                                 opts=dict(legend=[split_name],
                                                           title=var_name,
                                                           xlabel=xlabel,
                                                           ylabel=var_name))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]),  
                          win=self.plots[var_name], name=split_name)

    def draw(self, var_name, images):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(images)
        else:
            self.viz.images(images, win=self.plots[var_name])


class Logger(object):
    def __init__(self):
        self.plotter = VisdomPlotter()
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []

    '''
        Prints and saves the discriminator and generator loss for each batch
        in an epoch.
    '''
    def log_iteration_gan(self, epoch, d_loss, g_loss, real_score, fake_score):
        print("Epoch: %d, d_loss= %f, g_loss= %f, D(X)= %f, D(G(X))= %f" % (
              epoch, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), 
              real_score.data.cpu().mean(), fake_score.data.cpu().mean()))
        self.hist_D.append(d_loss.data.cpu().mean())
        self.hist_G.append(g_loss.data.cpu().mean())
        self.hist_Dx.append(real_score.data.cpu().mean())
        self.hist_DGx.append(fake_score.data.cpu().mean())

    '''
        Plots the discriminator and generator loss for each batch
        in an epoch as well as the scores of the discriminator and
        generator networks.
    '''
    def plot_epoch_w_scores(self, epoch):
        self.plotter.plot('Discriminator', 'train', epoch, 
                          np.array(self.hist_D).mean())
        self.plotter.plot('Generator', 'train', epoch, 
                          np.array(self.hist_G).mean())
        self.plotter.plot('D(X)', 'train', epoch, np.array(self.hist_Dx).
                                                                 mean())
        self.plotter.plot('D(G(X))', 'train', epoch, 
                          np.array(self.hist_DGx).mean())
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []

    '''
        Plots a set of generated images corresponding a right input sentence.
        Also plots a set of right images corresponding to the same set of 
        right input sentence. 
    '''
    def draw(self, right_images, fake_images):
        self.plotter.draw('generated images', 
                          fake_images.data.cpu().numpy()[:64] * 128 + 128)
        self.plotter.draw('real images', 
                          right_images.data.cpu().numpy()[:64] * 128 + 128)