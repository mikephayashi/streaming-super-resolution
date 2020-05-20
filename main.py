import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt
from functools import partial, lru_cache
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
import skimage.io as io

from VQVAE2 import VQVAE

import time
import torch.optim as optim
from torch.utils import data
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
import os
import torchvision


def load_model(model, checkpoint, device):
    # ckpt = torch.load(checkpoint)
    ckpt = torch.load(checkpoint, map_location='cpu')

    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE()

    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model


filename = './res/output.jpg'
vqvae_path = './vae example/vqvae_560.pt'

NUM_EPOCHS = 100
BATCH_SIZE = 128

if not os.path.exists("./params"):
    os.makedirs("./params")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('vqvae', vqvae_path, device)
# for param in model.parameters():
#     param.requires_grad = False
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# transform = torchvision.transforms.Compose([
# transforms.ToPILImage(),
# transforms.Resize(256),
# transforms.CenterCrop(256),
# transforms.ToTensor(),
# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])

transform = torchvision.transforms.Compose([
    transforms.ToTensor()
])
train_set = torchvision.datasets.ImageFolder(
    root="./res/resized",
    transform=transform
)
train_loader = data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

start = time.time()
cur_epochs = 0
total_epochs = []
losses = []
ssims = []
psnrs = []
param_count = 0
for epoch in range(NUM_EPOCHS):
    loss = 0
    ssim_score = 0
    psnr = 0

    for batch_features in train_loader:

        # reshape mini-batch data to [N, d] matrix
        # batch_features = batch_features[0].view(-1, DIMENSIONS).to(device)
        batch_features = batch_features[0].to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = criterion(outputs[0], batch_features[0])
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
        # SSIM
        ssim_score += ssim(batch_features.reshape(
            (-1, 3, 256, 256)), outputs.reshape((-1, 3, 256, 256)))

        # PSNR
        mse = torch.mean((batch_features.reshape((-1, 3, 256, 256)
                                                 ) - outputs.reshape((-1, 3, 256, 256))) ** 2)
        psnr += 20 * torch.log10(255.0 / torch.sqrt(mse))

    loss = loss / len(train_loader)
    ssim_score = ssim_score / len(train_loader)
    psnr = psnr / len(train_loader)
    with open("./params/losses.txt", "a") as file:
        file.write("{loss}, ".format(loss=loss))
        file.write("{ssim}, ".format(ssim=ssim_score))
        file.write("{psnr}\n".format(psnr=psnr))
    cur_epochs += 1
    total_epochs.append(cur_epochs)
    print("epoch : {}/{}, loss = {:.6f}, ssim = {}, psnr = {}".format(epoch +
                                                                      1, NUM_EPOCHS, loss, ssim_score, psnr))

    param_count += 1
    torch.save(model.state_dict(),
               "./params/params{num}.pt".format(num=param_count))


end = time.time()
print("Time: ", end - start)
