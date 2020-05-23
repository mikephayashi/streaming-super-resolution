import os
import sys
import time
import getopt
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


MODEL_NAME = None

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "n:", ["name="])
except getopt.GetoptError:
    print("Add model name")

for opt, arg in opts:
    if opt in ("-n", "--name"):
        MODEL_NAME = arg

if MODEL_NAME == None or (MODEL_NAME != "VAE" and MODEL_NAME != "VQVAE"):
    print("Add model name")
    sys.exit(0)

NUM_EPOCHS = 10
BATCH_SIZE = 64
SIZE = 128

if not os.path.exists("./params/VQVAE"):
    os.makedirs("./params/VQVAE")
if not os.path.exists("./logs/VQVAE"):
    os.makedirs("./logs/VQVAE")


print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if MODEL_NAME == "VAE":
    from models.VAE import VAE
    model = VAE()
elif MODEL_NAME == "VQVAE":
    from models.VQVAE2 import VQVAE
    model = VQVAE()
model.load_state_dict(torch.load("./params/{model_name}/params10.pt".format(model_name=MODEL_NAME)))
model = model.to(device)
model.eval()
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
criterion = nn.MSELoss()
count = 0

criterion = nn.MSELoss()


transform = torchvision.transforms.Compose([
    transforms.Resize(SIZE),
    transforms.CenterCrop(SIZE),
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
            ssim_score += ssim(batch_features.view((-1, 3, SIZE, SIZE)), outputs[0].view((-1, 3, SIZE, SIZE)))

            # PSNR
            mse = torch.mean((batch_features.view((-1, 3, SIZE, SIZE)
                                                ) - outputs[0].view((-1, 3, SIZE, SIZE))) ** 2)
            psnr += 20 * torch.log10(255.0 / torch.sqrt(mse))

            iteration += 1
            metric_counter += 1

            if iteration % 10 == 0:
                print("Iteration {it}".format(it=iteration))
                print(test_loss.item())
                end = time.time()
                time_dif = end - start
                print("Time: ", time_dif)
                test_loss = test_loss / metric_counter
                ssim_score = ssim_score / metric_counter
                psnr = psnr / metric_counter
                print("Loss: ", test_loss)
                print("SSIM: ", ssim_score)
                print("PSNR: ", psnr)
                with open("./logs/{model_name}/metrics.txt".format(model_name=MODEL_NAME), "a") as file:
                    file.write("loss: {loss}, ssim: {ssim}, psnr: {psnr}\n".format(loss=test_loss,ssim=ssim_score,psnr=psnr))
                test_loss = 0
                ssim_score = 0
                psnr = 0
                metric_counter = 0
                start = time.time()


        cur_epoch += 1
        print("Epoch:{cur_epoch}".format(cur_epoch=cur_epoch))