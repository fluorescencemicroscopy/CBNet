import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import skimage.io as skio
from conv_gamma import ConvGamma, train

def main():
    N_EPOCHS = 200
    BATCH_SIZE = 32
    AMPLITUDE = 4095

    data_path = #put path to data folder here
    images = #read in image training data here
    rlabels = #put labels here
    labels = np.zeros((rlabels.shape[0],2))
    labels[:,0] = rlabels
    nframes, nrows, ncols = images.shape
    print("images shape : ", images.shape)
    print("labels shape : ", labels.shape)
    images = np.expand_dims(images.astype(np.float32), axis = 3)
    images = np.transpose(images, axes=(0, 3, 1, 2))
    images /= AMPLITUDE
    labels /= 50000
    labels = labels.astype(np.float32)
    x_train, x_validation, y_train, y_validation = train_test_split(images,
            labels, test_size = 0.1, random_state = 0)
    model = ConvGamma()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ',device)
    train_losses, validation_losses = train(model, (x_train, y_train),
        (x_validation, y_validation), batch_size = BATCH_SIZE, epochs=N_EPOCHS,
        device=device, path_models='convgamma_models')

    np.save('train_losses_cgamma.npy', train_losses)
    np.save('validation_losses_cgamma.npy',validation_losses)

if __name__ == "__main__":
    main()
