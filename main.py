import time

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
import numpy as np

from Autoencoder import Autoencoder

NUM_EPOCHS = 5
BATCH_SIZE = 128


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


"""
Display Image
"""


def disp_img(input):
    grid = torchvision.utils.make_grid(input.cpu())
    img = grid / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


"""
Set up model
"""
print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(input_shape=100*100).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

"""
Load in data
"""
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set = torchvision.datasets.ImageFolder(
    root="./res/resized",
    transform=transform
)
train_loader = data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

"""
Train
"""
start = time.time()
cur_epochs = 0
total_epochs = []
losses = []
for epoch in range(NUM_EPOCHS):
    loss = 0
    # count = 0
    total_ssim = 0
    total_psnr = 0
    for batch_features in train_loader:

        # print("--------------------")
        # print("Batch features")
        # disp_img(batch_features[0])

        # reshape mini-batch data to [N, d] matrix
        batch_features = batch_features[0].view(-1, 100*100).to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = criterion(outputs, batch_features[0])
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

        # # SSIM
        # ssim_score = ssim(batch_features.reshape(
        #     (-1, 3, 100, 100)), outputs.reshape((-1, 3, 100, 100)))

        # # PSNR
        # mse = torch.mean((batch_features.reshape((-1, 3, 100, 100)
        #                                          ) - outputs.reshape((-1, 3, 100, 100))) ** 2)
        # psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))

        # print("Output")
        # disp_img(outputs.reshape([128, 3, 100, 100]))
        # print("--------------------")

    #     count += 1
    #     total_ssim += ssim_score
    #     total_psnr += psnr

    # print("ssim: ", total_ssim / count)
    # print("psnr: ", total_psnr / count)
    loss = loss / len(train_loader)
    losses.append(loss)
    cur_epochs += 1
    total_epochs.append(cur_epochs)
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, NUM_EPOCHS, loss))


end = time.time()
print("Time: ", end - start)

"""
Save model
"""
# Ref: https://pytorch.org/tutorials/beginner/saving_loading_models.html
torch.save(model.state_dict(), "./params.pt")
print("Saved")

# model = Autoencoder(input_shape=784)
# model.load_state_dict(torch.load("./params"))
# model.eval()
# print("Loaded")

"""
Loss Curve
"""
plt.scatter(total_epochs, losses)
plt.show()
