import os
import time
from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.optim as optim
from torch.utils import data
from functools import partial, lru_cache
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
import skimage.io as io
import matplotlib.pyplot as plt

from VQVAE2 import VQVAE


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
    # model.eval()

    return model


filename = './res/output.jpg'
vqvae_path = './vae example/vqvae_560.pt'

NUM_EPOCHS = 100
BATCH_SIZE = 128

if not os.path.exists("./params/VQVAE"):
    os.makedirs("./params/VQVAE")
if not os.path.exists("./logs/VQVAE"):
    os.makedirs("./logs/VQVAE")

print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('vqvae', vqvae_path, device)
count = 0
for param in model.parameters():
    if count != 58:
        param.requires_grad = False

print(count)
import pdb; pdb.set_trace()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()


transform = torchvision.transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
train_set = torchvision.datasets.ImageFolder(
    root="./res/frames/train",
    transform=transform
)
train_loader = data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

print("Number of batches {num_batches}".format(num_batches=len(train_loader)))

cur_epochs = 0
param_count = 0
total_epochs = []
losses = []
ssims = []
psnrs = []
start = time.time()
for epoch in range(NUM_EPOCHS):
    loss = 0
    ssim_score = 0
    psnr = 0
    iteration = 0

    for batch_features in train_loader:

        if iteration == 10:
            print("Iteration {it}".format(it=iteration))
            end = time.time()
            time_dif = end - start
            print("Time: ", time_dif)
        if iteration == 50:
            param_count += 1
            torch.save(model.state_dict(),
                       "./params/VQVAE/params{num}.pt".format(num=param_count))
            with open("./logs/VQVAE/times.txt", "a") as file:
                file.write("{time}".format(time=time_dif))
                file.close()
            start = time.time()

        batch_features = batch_features[0].to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = criterion(outputs[0], batch_features)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

        iteration += 1

    loss = loss / len(train_loader)
    ssim_score = ssim_score / len(train_loader)
    psnr = psnr / len(train_loader)
    with open("./logs/VQVAE/params.txt", "a") as file:
        file.write("{loss}, ".format(loss=loss))
        file.write("{ssim}, ".format(ssim=ssim_score))
        file.write("{psnr}\n".format(psnr=psnr))
    cur_epochs += 1
    total_epochs.append(cur_epochs)
    print("epoch : {}/{}, loss = {:.6f}, ssim = {}, psnr = {}".format(epoch +
                                                                      1, NUM_EPOCHS, loss, ssim_score, psnr))
