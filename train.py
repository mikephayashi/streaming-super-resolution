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

import torch.multiprocessing as mp

from models.VQVAE2 import VQVAE

NUM_EPOCHS = 10
BATCH_SIZE = 128

if not os.path.exists("./params/VQVAE"):
    os.makedirs("./params/VQVAE")
if not os.path.exists("./logs/VQVAE"):
    os.makedirs("./logs/VQVAE")

print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE()
model.load_state_dict(torch.load("./vae example/vqvae_560.pt"))
model.to(device)
count = 0

optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
criterion = nn.MSELoss()


transform = torchvision.transforms.Compose([
    transforms.Resize(360),
    transforms.CenterCrop(360),
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
start = time.time()
for epoch in range(NUM_EPOCHS):
    iteration = 0
    total_loss = 0
    train_loss = 0
    loss_count = 0

    import pdb; pdb.set_trace()
    for batch_features in train_loader:

        # Loss and back
        batch_features = batch_features[0].to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = criterion(outputs[0], batch_features)
        train_loss.backward()
        optimizer.step()
        total_loss += train_loss.item()

        # Update counters
        iteration += 1
        loss_count += 1

        # Periodically print and save
        if iteration % 10 == 0:
            print("Iteration {it}".format(it=iteration))
            print("loss:", total_loss / loss_count)
            end = time.time()
            time_dif = end - start
            print("Time: ", time_dif)
        if iteration % 50 == 0:
            param_count += 1
            torch.save(model.state_dict(),
                       "./params/VQVAE/params{num}.pt".format(num=param_count))
            total_loss = total_loss / loss_count
            with open("./logs/VQVAE/params.csv", "a") as file:
                file.write("{train_loss},".format(train_loss=train_loss.item()))
            start = time.time()
            total_loss = 0
            loss_count = 0

    cur_epochs += 1
    print("Epoch:{epoch}".format(epoch = cur_epochs))