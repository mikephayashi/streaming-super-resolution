"""
https://github.com/sksq96/pytorch-vae/blob/master/vae.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# class Break(nn.Module):

#     def forward(self, input):
#         import pdb
#         pdb.set_trace()
#         return input


class UnFlatten(nn.Module):
    def forward(self, input, size=102400):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=102400, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=5),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=4),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar

# """
# https://github.com/coolvision/vae_conv/blob/master/mvae_conv_model.py
# """

# from __future__ import print_function
# import argparse
# import torch
# import torch.utils.data
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
# from torchvision import datasets, transforms

# import os
# import random
# import torch.utils.data
# import torchvision.utils as vutils
# import torch.backends.cudnn as cudnn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ngf = 64
# ndf = 64
# nc = 3

# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()

#         nz = 64
#         self.have_cuda = False
#         self.nz = nz

#         self.encoder = nn.Sequential(
#             # input is (nc) x 28 x 28
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 14 x 14
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 7 x 7
#             nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 4 x 4
#             nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
#             # nn.BatchNorm2d(1024),
#             nn.LeakyReLU(0.2, inplace=True),
#             # nn.Sigmoid()
#         )

#         self.decoder = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(     1024, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(ngf * 2,     nc, 6, 8, 1, bias=False),
#             # nn.BatchNorm2d(ngf),
#             # nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
#             # nn.Tanh()
#             nn.Sigmoid()
#             # state size. (nc) x 64 x 64
#         )

#         self.fc1 = nn.Linear(1024, 512)
#         self.fc21 = nn.Linear(512, nz)
#         self.fc22 = nn.Linear(512, nz)

#         self.fc3 = nn.Linear(nz, 512)
#         self.fc4 = nn.Linear(512, 1024)

#         self.lrelu = nn.LeakyReLU()
#         self.relu = nn.ReLU()
#         # self.sigmoid = nn.Sigmoid()

#     def encode(self, x):
#         conv = self.encoder(x);
#         # print("encode conv", conv.size())
#         h1 = self.fc1(conv.view(-1, 1024))
#         # print("encode h1", h1.size())
#         return self.fc21(h1), self.fc22(h1)

#     def decode(self, z):
#         h3 = self.relu(self.fc3(z))
#         deconv_input = self.fc4(h3)
#         # print("deconv_input", deconv_input.size())
#         deconv_input = deconv_input.view(-1,1024,1,1)
#         # print("deconv_input", deconv_input.size())
#         return self.decoder(deconv_input)

#     def reparametrize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         if self.have_cuda:
#             eps = torch.cuda.FloatTensor(std.size()).normal_()
#         else:
#             eps = torch.FloatTensor(std.size()).normal_()
#         eps = Variable(eps).to(device)
#         return eps.mul(std).add_(mu)

#     def forward(self, x):
#         # print("x", x.size())
#         mu, logvar = self.encode(x)
#         # print("mu, logvar", mu.size(), logvar.size())
#         z = self.reparametrize(mu, logvar)
#         # print("z", z.size())
#         decoded = self.decode(z)
#         # print("decoded", decoded.size())
#         return decoded, mu, logvar