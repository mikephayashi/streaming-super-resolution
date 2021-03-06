import os
import time
from math import isnan
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

from models.AE import AE



NUM_EPOCHS = 10
BATCH_SIZE = 128
DIMENSIONS = 128

if not os.path.exists("./params/AE"):
    os.makedirs("./params/AE")
if not os.path.exists("./logs/AE"):
    os.makedirs("./logs/AE")


print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE(input_shape=DIMENSIONS)
model.load_state_dict(torch.load('./params/AE/params8.pt'))
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
start = time.time()
for epoch in range(NUM_EPOCHS):
    iteration = 0
    total_loss = 0
    train_loss = 0
    loss_count = 0

    for batch_features in train_loader:

        # Loss and back
        batch_features = batch_features[0].to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = criterion(outputs, batch_features)
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
                    "./params/AE/params{num}.pt".format(num=param_count))
            total_loss = total_loss / loss_count
            with open("./logs/AE/params.csv", "a") as file:
                file.write("{train_loss},".format(train_loss=train_loss.item()))
            start = time.time()
            total_loss = 0
            loss_count = 0

    cur_epochs += 1
    print("Epoch:{epoch}".format(epoch = cur_epochs))
