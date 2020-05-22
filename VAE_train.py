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

from VAE import VAE





NUM_EPOCHS = 100
BATCH_SIZE = 128

if not os.path.exists("./params"):
    os.makedirs("./params")
if not os.path.exists("./params/VAE"):
    os.makedirs("./params/VAE")
if not os.path.exists("./logs/VAE"):
    os.makedirs("./logs/VAE")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
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
        if iteration == 50:
            param_count += 1
            torch.save(model.state_dict(),
                       "./params/VAE/params{num}.pt".format(num=param_count))
            end = time.time()
            time_dif = end - start
            print("Time: ", time_dif)
            with open("./logs/VAE/times.txt", "a") as file:
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
    with open("./logs/VAE/params.txt", "a") as file:
        file.write("{loss}, ".format(loss=loss))
        file.write("{ssim}, ".format(ssim=ssim_score))
        file.write("{psnr}\n".format(psnr=psnr))
    cur_epochs += 1
    total_epochs.append(cur_epochs)
    print("epoch : {}/{}, loss = {:.6f}, ssim = {}, psnr = {}".format(epoch +
                                                                      1, NUM_EPOCHS, loss, ssim_score, psnr))
