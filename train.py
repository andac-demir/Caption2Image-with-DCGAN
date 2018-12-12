import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from txt2image_dataset import Text2ImageDataset
from model import dcgan
from utils import Utils
from visualize import Logger
from PIL import Image
import os

class Train(object):
    def __init__(self, dataset, split, lr, l1_coef, l2_coef, 
                 batch_size, num_workers, epochs):
        self.generator = torch.nn.DataParallel(dcgan.generator().cuda())
        self.discriminator = torch.nn.DataParallel(dcgan.discriminator().
                                                                 cuda())
        self.generator.apply(Utils.weights_init)
        self.discriminator.apply(Utils.weights_init)
        self.filename = dataset 
        
        if dataset == 'birds':
            self.dataset = Text2ImageDataset('Datasets/birds.hdf5', 
                                             split=split)
        elif dataset == 'flowers':
            self.dataset = Text2ImageDataset('Datasets/flowers.hdf5', 
                                             split=split)
        else:
            print('Dataset not available, select either birds or flowers.')
            exit()

        self.noise_dim = 100
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.beta1 = 0.5
        self.num_epochs = epochs
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, 
                                      shuffle=True, 
                                      num_workers=self.num_workers)
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr,
                                       betas=(self.beta1, 0.999))
        self.optimD = torch.optim.Adam(self.discriminator.parameters(), 
                                       lr=self.lr, betas=(self.beta1, 0.999))
        self.logger = Logger()
        self.checkpoints_path = 'checkpoints'

    '''
        Saves the models to the directory Model
    '''
    def saveModel(self, generator, discriminator):
        torch.save(generator.state_dict(), f="TrainedModels/generator_%s.model"
                                                               %self.filename)
        torch.save(discriminator.state_dict(), 
                   f="TrainedModels/discriminator_%s.model"%self.filename)
        print("Models saved successfully.")

    def train_network(self):
        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        iteration = 0
        for epoch in range(self.num_epochs):
            for sample in self.data_loader:
                iteration += 1             
                right_images = Variable(sample['right_images'].float()).cuda()
                right_embed = Variable(sample['right_embed'].float()).cuda()
                wrong_images = Variable(sample['wrong_images'].float()).cuda()
                
                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))
                # Smoothing prevents the discriminator from overpowering the
                # generator adding penalty when the discriminator is 
                # too confident - this avoids the generator lazily copying the 
                # images in the training data.
                smoothed_real_labels = Variable(torch.FloatTensor(Utils.
                                            smooth_label(real_labels.numpy(),
                                                              -0.1))).cuda()
                real_labels = Variable(real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()
                
                # Train the discriminator
                self.discriminator.zero_grad()
                outputs, activation_real = self.discriminator(right_images, 
                                                              right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs
                outputs, _ = self.discriminator(wrong_images, right_embed)
                wrong_loss = criterion(outputs, fake_labels)

                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, _ = self.discriminator(fake_images, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs
                d_loss = real_loss + fake_loss + wrong_loss
                d_loss.backward()
                self.optimD.step()
                
                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, activation_fake = self.discriminator(fake_images, 
                                                              right_embed)
                _, activation_real = self.discriminator(right_images, 
                                                        right_embed)
                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)
                # the first term in the loss function of the generator is the
                # regular cross entropy loss, the second term is 
                # feature matching loss which measures the distance 
                # between the real and generated images statistics 
                # by comparing intermediate layers activations and
                # the third term is L1 distance between the generated and 
                # real images, this is helpful for the conditional case
                # because it links the embedding feature vector directly 
                # to certain pixel values.
                g_loss = criterion(outputs, real_labels) \
                         + self.l2_coef * l2_loss(activation_fake, 
                                                  activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, right_images)
                g_loss.backward()
                self.optimG.step()
                if iteration % 5 == 0:
                    self.logger.log_iteration_gan(epoch,d_loss, g_loss, 
                                                  real_score, fake_score)
                    self.logger.draw(right_images, fake_images)

            self.logger.plot_epoch_w_scores(epoch)
            if (epoch) % 10 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, 
                                      self.checkpoints_path, epoch)
        
        # saves the trained model:
        self.saveModel(self.generator, self.discriminator)

    def predict(self):
        self.generator.load_state_dict(torch.load(
                                       ("TrainedModels/generator_%s.model" 
                                       %self.filename), map_location='cpu'))
        self.discriminator.load_state_dict(torch.load(
                                           ("TrainedModels/discriminator_%s"
                                           ".model"%self.filename), 
                                           map_location='cpu'))
        for sample in self.data_loader:
            if not os.path.exists('results/{0}'.format('')):
                os.makedirs('results/{0}'.format(''))

            right_images = Variable(sample['right_images'].float()).cuda()
            right_embed = Variable(sample['right_embed'].float()).cuda()
            # Train the generator
            noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = self.generator(right_embed, noise)
            self.logger.draw(right_images, fake_images)
            for image, t in zip(fake_images, sample['txt']):
                im = Image.fromarray(image.data.mul_(127.5).add_(127.5).\
                                 byte().permute(1, 2, 0).cpu().numpy())
                im.save('results/{0}/{1}.jpg'.format('', 
                                                     t.replace("/", "")[:100]))
                print(t)







