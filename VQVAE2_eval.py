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
    model.eval()

    return model


filename = './res/output.jpg'
vqvae_path = './params/VQVAE/params1.pt'

NUM_EPOCHS = 100
BATCH_SIZE = 128

if not os.path.exists("./params/VQVAE"):
    os.makedirs("./params/VQVAE")
if not os.path.exists("./logs/VQVAE"):
    os.makedirs("./logs/VQVAE")

print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('vqvae', vqvae_path, device)
model.share_memory()
count = 0
# for param in model.parameters():
#     count += 1
#     if count != 58:
#         param.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
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

    for batch_features in test_loader:

        batch_features = batch_features[0].to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        test_loss = criterion(outputs[0], batch_features)

        # SSIM
        ssim_score += ssim(batch_features.view(
            (-1, 3, 128, 128)), outputs[0].view((-1, 3, 128, 128)))

        # PSNR
        mse = torch.mean((batch_features.view((-1, 3, 128, 128)
                                              ) - outputs[0].view((-1, 3, 128, 128))) ** 2)
        psnr += 20 * torch.log10(255.0 / torch.sqrt(mse))

        print("SSIM: ", ssim_score)
        print("PSNR: ", psnr)

        if iteration % 10 == 0:
            print("Iteration {it}".format(it=iteration))
            print(test_loss.item())
            end = time.time()
            time_dif = end - start
            print("Time: ", time_dif)
        if iteration == 50:
            start = time.time()
                #         with open("./logs/VQVAE/params.txt", "a") as file:
                # file.write("{train_loss}".format(train_loss=train_loss.item()))

        iteration += 1

    print("Epoch:{loss}".format(loss=loss))
