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

from pytorch_msssim import ssim

from models.VQVAE2 import VQVAE



if not os.path.exists("./logs/VQVAE"):
    os.makedirs("./logs/VQVAE")

NUM_EPOCHS = 100
BATCH_SIZE = 128

if not os.path.exists("./params/VQVAE"):
    os.makedirs("./params/VQVAE")
if not os.path.exists("./logs/VQVAE"):
    os.makedirs("./logs/VQVAE")

print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE()
model.load_state_dict(torch.load("./params/VQVAE/params1.p"))
model.to(device)
count = 0

criterion = nn.MSELoss()


transform = torchvision.transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
test_set = torchvision.datasets.ImageFolder(
    root="./res/frames/test",
    transform=transform
)
test_loader = data.DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

print("Number of batches {num_batches}".format(num_batches=len(test_loader)))

cur_epoch = 0
start = time.time()
with torch.no_grad():

    for epoch in range(NUM_EPOCHS):
        test_loss = 0
        ssim_score = 0
        psnr = 0
        iteration = 0
        metric_counter = 0

        for batch_features in test_loader:

            batch_features = batch_features[0].to(device)
            outputs = model(batch_features)
            test_loss += criterion(outputs[0], batch_features)

            # SSIM
            ssim_score += ssim(batch_features.view((-1, 3, 128, 128)), outputs[0].view((-1, 3, 128, 128)))

            # PSNR
            mse = torch.mean((batch_features.view((-1, 3, 128, 128)
                                                ) - outputs[0].view((-1, 3, 128, 128))) ** 2)
            psnr += 20 * torch.log10(255.0 / torch.sqrt(mse))

            iteration += 1
            metric_counter += 1

            if iteration % 10 == 0:
                print("Iteration {it}".format(it=iteration))
                print(test_loss.item())
                end = time.time()
                time_dif = end - start
                print("Time: ", time_dif)
                print("Loss: ", test_loss / metric_counter)
                print("SSIM: ", ssim_score / metric_counter)
                print("PSNR: ", psnr / metric_counter)
            if iteration == 50:
                start = time.time()
                test_loss = test_loss / metric_counter
                ssim_score = ssim_score / metric_counter
                psnr = psnr / metric_counter
                with open("./logs/VQVAE/metrics.txt", "a") as file:
                    file.write("loss: {loss}, ssim: {ssim}, psnr: {psnr}".format(loss=test_loss,ssim=ssim_score,psnr=psnr))
                test_loss = 0
                ssim_score = 0
                psnr = 0
                metric_counter = 0

        cur_epoch += 1
        print("Epoch:{cur_epoch}".format(cur_epoch=cur_epoch))