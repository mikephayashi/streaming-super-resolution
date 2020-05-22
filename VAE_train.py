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

if not os.path.exists("./params/VAE"):
    os.makedirs("./params/VAE")
if not os.path.exists("./logs/VAE"):
    os.makedirs("./logs/VAE")


print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE()
model = model.to(device)
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

        batch_features = batch_features[0].to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = criterion(outputs[0], batch_features)
        train_loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            print("Iteration {it}".format(it=iteration))
            end = time.time()
            time_dif = end - start
            print("Time: ", time_dif)
        if iteration == 50:
            param_count += 1
            torch.save(model.state_dict(),
                       "./params/VAE/params{num}.pt".format(num=param_count))
            with open("./logs/VAE/params.txt", "a") as file:
                file.write("{train_loss}".format(loss=train_loss.item()))
            start = time.time()

        iteration += 1

    print("Epoch:{loss}".format(loss=loss))
