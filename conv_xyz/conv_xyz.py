import os

import numpy as np
# import progressbar as pb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


class ConvXYZ(nn.Module):

    def __init__(
        self,
        nrows = 32,
        ncols = 32,
        in_channels = 1,
        out_channels = 3,
        conv_dim = 64,
        drop_prob = 0.2,
        alpha = 0.1
        ):
        super(ConvXYZ, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_dim = conv_dim
        self.drop_prob = drop_prob
        self.alpha = alpha

        # define computational layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.conv_dim, (3, 3), stride = (1, 1), padding = 1, bias=False),
            nn.BatchNorm2d(self.conv_dim),
            nn.LeakyReLU(self.alpha)
        )

        self.conv_stride1 = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, (2, 2), stride = (2, 2), padding = 0, bias = False),
            nn.BatchNorm2d(self.conv_dim),
            nn.LeakyReLU(self.alpha)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim*2, (3, 3), stride = (1, 1), padding = 1, bias=False),
            nn.BatchNorm2d(self.conv_dim*2),
            nn.LeakyReLU(self.alpha)
        )

        self.conv_stride2 = nn.Sequential(
            nn.Conv2d(self.conv_dim*2, self.conv_dim*2, (2, 2), stride = (2, 2), padding = 0, bias = False),
            nn.BatchNorm2d(self.conv_dim*2),
            nn.LeakyReLU(self.alpha)
        )

        self.flattend_size = (nrows//4) * (ncols//4) * self.conv_dim*2
        self.fc1 = nn.Sequential(
            nn.Linear(self.flattend_size, self.conv_dim*2),
            nn.BatchNorm1d(self.conv_dim*2),
            nn.LeakyReLU(self.alpha),
            nn.Dropout(self.drop_prob)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.conv_dim*2, self.conv_dim),
            nn.BatchNorm1d(self.conv_dim),
            nn.LeakyReLU(self.alpha),
            nn.Dropout(self.drop_prob)
        )

        self.fc3 = nn.Linear(self.conv_dim, 3)

    def forward(
        self,
        x
        ):
        x = self.conv1(x)
        x = self.conv_stride1(x)
        x = self.conv2(x)
        x = self.conv_stride2(x)
        x = x.view(-1, self.flattend_size)
        x = self.fc1(x)
        x = self.fc2(x)

        return self.fc3(x)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = np.sqrt(2/n)
        m.weight.data.normal_(0, y)
        m.bias.data.fill_(0)

    elif isinstance(m, nn.Conv2d):
        n = m.kernel_size[0]
        y = np.sqrt(1/n)
        m.weight.data.normal_(0, y)


def train(
    model,
    train_data,      # Tuple (images, labels)
    validation_data,
    batch_size = 32,
    n_epochs = 2,
    path_models = "convxyz_models",
    device = "cpu"
    ):
    if not (isinstance(train_data, tuple) and len(train_data) == 2):
        raise TypeError("train_data is a tuple of images and labels")

    if not os.path.isdir(path_models):
        os.mkdir(path_models)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 10, min_lr = 0.000000001, verbose = False)

    train_x, train_y = train_data
    validation_x, validation_y = validation_data

    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    train_dataset = Data.TensorDataset(train_x, train_y)

    train_loader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 2
    )

    validation_x = torch.from_numpy(validation_x)
    validation_y = torch.from_numpy(validation_y)
    validation_dataset = Data.TensorDataset(validation_x, validation_y)

    validation_loader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 2
    )

    model.to(device)

    #widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    #timer = pb.ProgressBar(widget=widget, maxval=n_epochs).start()

    train_losses = []
    valid_losses = []
    optimal_validation_loss = np.inf
    for i in range(n_epochs):
        epoch_train_losses = []
        model.train()
        for x, y in train_loader:
            x = x.requires_grad_(True).to(device)
            y = y.requires_grad_(True).to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.cpu().detach().item())

        epoch_train_loss = np.mean(epoch_train_losses)
        train_losses.append(epoch_train_loss)

        epoch_validation_losses = []
        model.eval()
        with torch.no_grad():
            for x, y in validation_loader:
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)
                epoch_validation_losses.append(loss.cpu().detach().item())

        epoch_validation_loss = np.mean(epoch_validation_losses)
        valid_losses.append(epoch_validation_loss)
        scheduler.step(epoch_validation_loss)

        print("\nEpoch {}, Training Loss: {:.5f},\t Validation Loss : {:.5f}".format(i, epoch_train_loss, epoch_validation_loss))

        # save models if validation loss is improved.
        if (epoch_validation_loss < optimal_validation_loss):
            print("Validation loss is improved: {} from {}".format(epoch_validation_loss, optimal_validation_loss))
            optimal_validation_loss = epoch_validation_loss
            torch.save(model.state_dict(), os.path.join(path_models, "final_model.pt"))
            print("")

        #timer.update(i + 1)

    return train_losses, valid_losses

def predict(
    model,
    x
    ):

    # if isinstance(x, np.ndarray):
    #     x = torch.FloatTensor(x)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    prediction = []
    for i in range(0, len(x), 8):
        try:
            tmp_x = torch.from_numpy(x[i:i+8]).float().to(device)
        except:
            tmp_x = torch.from_numpy(x[i:]).float().to(device)

        tmp_y = model(tmp_x)
        v = tmp_y.detach().cpu().numpy()
        prediction.append(v)
        del tmp_x, tmp_y

    prediction = np.vstack(prediction)

    return prediction
