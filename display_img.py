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

def disp_img(input):
    grid = torchvision.utils.make_grid(input.cpu())
    img = grid / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


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


vqvae_path = './params/VQVAE/params4.pt'

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
model = load_model('vqvae', vqvae_path, device)
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

            print("Loss, ", test_loss)
            print("ssim ", ssim_score)
            print("psnr ", psnr)
            disp_img(batch_features[0])
            disp_img(outputs[0])


        cur_epoch += 1
        print("Epoch:{cur_epoch}".format(cur_epoch=cur_epoch))