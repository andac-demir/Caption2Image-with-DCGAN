'''
    Based on DC-GAN:
    https://github.com/pytorch/examples/blob/master/dcgan/main.py
    
    Network Architecture:
    First, the text query t is encoded using a fully connected layer to a 
    small dimension (128). 
    Then concatenated to the noise vector z.  
    Following this, concatenated vector is projected to a small spatial extent
    convolutional representation with many feature maps.
    A series of four fractionally-strided convolutions convert this high level
    representation into a 64 x 64 pixel image. 
    No fully connected or pooling layers used.

    Architecture Guidelines:
    -Pooling layers replaced with fractional strided convolutions (generator)
     and strided convolutions (discriminator).
    -Batchnorm used in both the generator and discriminator.
    -Fully connected layers removed for increased depth.
    -In generator ReLU activation used, except for the output which uses Tanh.
    -In discriminator Leaky ReLU activation used for all layers.
'''
import torch
import torch.nn as nn
from torch.autograd import Variable

class generator(nn.Module):
	def __init__(self):
		super(generator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.noise_dim = 100
		self.embed_dim = 1024
		self.projected_embed_dim = 128
		self.latent_dim = self.noise_dim + self.projected_embed_dim
		self.ngf = 64
		self.projection = nn.Sequential(
			nn.Linear(in_features=self.embed_dim, 
                      out_features=self.projected_embed_dim),
			nn.BatchNorm1d(num_features=self.projected_embed_dim),
			nn.LeakyReLU(negative_slope=0.2, inplace=True)
			)
		self.genNet = nn.Sequential(
			nn.ConvTranspose2d(in_channels=self.latent_dim, 
                               out_channels=self.ngf * 8, 
                               kernel_size=4, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(num_features=self.ngf * 8),
			nn.ReLU(inplace=True),
            # 4 x 4 x (ngf x 8)
			nn.ConvTranspose2d(in_channels=self.ngf * 8, 
                               out_channels=self.ngf * 4, 
                               kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=self.ngf * 4),
			nn.ReLU(inplace=True),
            # 8 x 8 x (ngf x 4)
			nn.ConvTranspose2d(in_channels=self.ngf * 4, 
                               out_channels=self.ngf * 2, 
                               kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=self.ngf * 2),
			nn.ReLU(inplace=True),
            # 16 x 16 x (ngf x 2)
			nn.ConvTranspose2d(in_channels=self.ngf * 2, 
                               out_channels=self.ngf, 
                               kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=self.ngf),
			nn.ReLU(inplace=True),
            # 32 x 32 x (ngf)
			nn.ConvTranspose2d(in_channels=self.ngf, 
                               out_channels=self.num_channels, 
                               kernel_size=4, stride=2, padding=1, bias=False),
			nn.Tanh()
            # 64 x 64 x num_channels
			)

	def forward(self, embed_vector, z):
		projected_embed = self.projection(embed_vector).unsqueeze(2).\
                                                        unsqueeze(3)
		latent_vector = torch.cat([projected_embed, z], 1)
		output = self.genNet(latent_vector)
		return output


class discriminator(nn.Module):
	def __init__(self):
		super(discriminator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.embed_dim = 1024
		self.projected_embed_dim = 128
		self.ndf = 64
		self.B_dim = 128
		self.C_dim = 16
		self.discNet1 = nn.Sequential(
			# input is 64 x 64 x (num_channels)
			nn.Conv2d(in_channels=self.num_channels, out_channels=self.ndf, 
                      kernel_size=4, stride=2, padding=1, bias=False),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			# 32 x 32 x (ndf)
			nn.Conv2d(in_channels=self.ndf, 
                      out_channels=self.ndf * 2, 
                      kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=self.ndf * 2),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			# 16 x 16 x (ndf*2)
			nn.Conv2d(in_channels=self.ndf * 2, out_channels=self.ndf * 4, 
                      kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=self.ndf * 4),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			# 8 x 8 x (ndf*4)
			nn.Conv2d(in_channels=self.ndf * 4, out_channels=self.ndf * 8, 
                      kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(num_features=self.ndf * 8),
			nn.LeakyReLU(negative_slope=0.2, inplace=True))
		self.projector = concatEmbed(self.embed_dim, self.projected_embed_dim)
		self.discNet2 = nn.Sequential(
			# 4 x 4 x (ndf*8)
			nn.Conv2d(in_channels=self.ndf * 8 + self.projected_embed_dim, 
                      out_channels=1, kernel_size=4, stride=1, padding=0, 
                      bias=False),
			nn.Sigmoid())	

	def forward(self, inp, embed):
		x_intermediate = self.discNet1(inp)
		x = self.projector(x_intermediate, embed)
		x = self.discNet2(x)
		return x.view(-1, 1).squeeze(1), x_intermediate

'''
    Encodes the text input for the discriminator network.
    Replicates this embedding vector spatially to match 
    the spatial dimensions of the visual feature representation tensor
    so they can be concatenated. 
    Provides depth concatenation before the final convolutional layer.
'''
class concatEmbed(nn.Module):
    def __init__(self, embed_dim, projected_embed_dim):
        super(concatEmbed, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=projected_embed_dim),
            nn.BatchNorm1d(num_features=projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, inp, embed):
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(4, 4, 1, 1).\
                                          permute(2, 3, 0, 1)
        hidden_concat = torch.cat([inp, replicated_embed], 1)
        return hidden_concat