import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import skimage.io as skio


class ConvGamma(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=3,padding=1,bias=False)
        self.conv1_1 = nn.Conv2d(64,64,kernel_size=3,padding=1,bias=False)
        self.act1 = nn.LeakyReLU(.1)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(64,16,kernel_size=3,padding=1,bias=False)
        self.conv2_1 = nn.Conv2d(16,16,kernel_size=3,padding=1,bias=False)
        self.act2 = nn.LeakyReLU(.1)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(16,16,kernel_size=3,padding=1,bias=False)
        self.act3 = nn.LeakyReLU(.1)
        self.pool3 = nn.AvgPool2d(2)
        self.step1 = nn.Linear(16*4*4,256)
        self.act = nn.LeakyReLU(.15)
        self.step2 = nn.Linear(256,2)

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.pool1(self.act1(self.conv1_1(out)))
        out = self.act2(self.conv2(out))
        out = self.pool2(self.act2(self.conv2_1(out)))
        out_inter = out
        out = self.pool3(self.act3(self.conv3(out))+out_inter)
        out = out.reshape(-1,16*4*4)
        out = self.act(self.step1(out))
        out = self.step2(out)
        return out

def ugloss(y_pred, y_true):
    mu = y_pred[:,0]
    sigma = torch.exp(y_pred[:,1])
    cond_dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)
    loss = -cond_dist.log_prob(y_true[:,0])
    return torch.mean(loss, axis=-1)

def train(model,
          train_data,
          validation_data,
          batch_size=32,
          epochs=10,
          device="cpu",
          path_models = "convgamma_models"):
    if not (isinstance(train_data, tuple) and len(train_data) == 2):
        raise TypeError("train_data is a tuple of images and labels")

    if not os.path.isdir(path_models):
        os.mkdir(path_models)

    loss_fxn = ugloss#nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_ims, train_labels = train_data
    validation_ims, validation_labels = validation_data

    train_ims = torch.from_numpy(train_ims)
    train_labels = torch.from_numpy(train_labels)
    train_dataset = TensorDataset(train_ims, train_labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True)

    validation_ims = torch.from_numpy(validation_ims)
    validation_labels = torch.from_numpy(validation_labels)
    validation_dataset = TensorDataset(validation_ims, validation_labels)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                   shuffle=True)
    model.to(device)

    train_losses = []
    validation_losses  =[]
    best_validation_loss = np.inf
    for i in range(epochs):
        trainloss_thisepoch = []
        model.train()
        for tlim, tllab in train_loader:
            tlim = tlim.requires_grad_(True).to(device)
            tllab = tllab.requires_grad_(True).to(device)
            out = model(tlim)
            loss = loss_fxn(out, tllab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainloss_thisepoch.append(loss.cpu().detach().item())

        train_losses.append(np.mean(trainloss_thisepoch))

        valloss_thisepoch = []
        model.eval()
        with torch.no_grad():
            for vlim, vllab in validation_loader:
                vlim = vlim.to(device)
                vllab = vllab.to(device)
                out = model(vlim)
                loss = loss_fxn(out, vllab)
                valloss_thisepoch.append(loss.cpu().detach().item())

        validation_losses.append(np.mean(valloss_thisepoch))
        print("Epoch: {:>3d},  Train loss: {:>.5f},  Validation loss: {:>.5f}".
              format(i, train_losses[-1], validation_losses[-1]))

        if (validation_losses[-1] < best_validation_loss):
            print("Validation loss is improved: {} from {}".format(validation_losses[-1], best_validation_loss))
            best_validation_loss = validation_losses[-1]
            torch.save(model.state_dict(), os.path.join(path_models,"final_model.pt"))
            print("")

    return train_losses, validation_losses
