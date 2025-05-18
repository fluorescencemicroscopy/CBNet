import os
import glob

import pickle

import numpy as np

import imageio

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim


from conv_xyz import ConvXYZ, train, initialize_weights




def main():
    N_EPOCHS = 400
    BATCH_SIZE = 64
    AMPLITUDE = 4095.
    # Load and preprocess data
    data_path = #put data path here
    out_path = '.'
    path = os.path.join(out_path, "aug_output0")

    images = #load image data here
    labels = #load labels here
    nframes, nrows, ncols = images.shape
    print("images shape : ", images.shape)
    print("labels shape : ", labels.shape)

    print("max : {}, min : {}".format(np.max(images), np.min(images)))
    print("max labels: ", np.max(labels, axis = 0))
    print("min labels: ",np.min(labels, axis = 0))

    images = np.expand_dims(images.astype(np.float32), axis = 3)
    images = np.transpose(images, axes=(0, 3, 1, 2))
    images /= AMPLITUDE

    labels[:, :2] /= 32
    labels[:, 2]  /= 2.585

    labels = labels.astype(np.float32)
    print(images.shape)
    print(labels.shape)
    ### PREPROCESS DATA ###
    x_train, x_validation, y_train, y_validation = train_test_split(images, labels, test_size = 0.1, random_state = 0)
    print(x_train.shape, y_train.shape)
    print(x_validation.shape, y_validation.shape)


    model = ConvXYZ(nrows = nrows, ncols = ncols, in_channels = 1, out_channels = 3,
        conv_dim = 64, drop_prob = 0.1, alpha = 0.1)
    model.apply(initialize_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_losses, validation_losses = train(model, (x_train, y_train), (x_validation, y_validation), batch_size = BATCH_SIZE, n_epochs = N_EPOCHS, device = device)

    with open("train_losses.pk", "wb") as f:
        pickle.dump(train_losses, f)

    with open("validation_losses.pk", "wb") as f:
        pickle.dump(validation_losses, f)



if __name__ == "__main__":
    main()
