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
    def forward(self, input, size=12544):
        return input.view(input.size(0), size, 1, 1)


"""
128
"""
class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=12544, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=11, stride=8a),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=8, stride=2),
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
# class VAE(nn.Module):
#     def __init__(self, image_channels=3, h_dim=256, z_dim=32):
#         super(VAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=7, stride=3),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=2, stride=2),
#             nn.ReLU(),
#             Flatten()
#         )

#         self.fc1 = nn.Linear(h_dim, z_dim)
#         self.fc2 = nn.Linear(h_dim, z_dim)
#         self.fc3 = nn.Linear(z_dim, h_dim)

#         self.decoder = nn.Sequential(
#             UnFlatten(),
#             nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=5),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=5),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=5, stride=5),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=3),
#             nn.Sigmoid(),
#         )

#     def reparameterize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         # return torch.normal(mu, std)
#         esp = torch.randn(*mu.size()).to(device)
#         z = mu + std * esp
#         return z

#     def bottleneck(self, h):
#         mu, logvar = self.fc1(h), self.fc2(h)
#         z = self.reparameterize(mu, logvar)
#         return z, mu, logvar

#     def representation(self, x):
#         return self.bottleneck(self.encoder(x))[0]

#     def forward(self, x):
#         h = self.encoder(x)
#         z, mu, logvar = self.bottleneck(h)
#         z = self.fc3(z)
#         return self.decoder(z), mu, logvar

"""
360
"""
# class VAE(nn.Module):
#     def __init__(self, image_channels=3, h_dim=102400, z_dim=32):
#         super(VAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2),
#             nn.ReLU(),
#             Flatten()
#         )

#         self.fc1 = nn.Linear(h_dim, z_dim)
#         self.fc2 = nn.Linear(h_dim, z_dim)
#         self.fc3 = nn.Linear(z_dim, h_dim)

#         self.decoder = nn.Sequential(
#             UnFlatten(),
#             nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=5),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=5),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=5, stride=5),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=3),
#             nn.Sigmoid(),
#         )