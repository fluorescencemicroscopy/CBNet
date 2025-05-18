import os, sys

import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Normalizer

from torch.utils.data import DataLoader

from retinanet import csv_eval

assert int(torch.__version__.split('.')[0]) >= 1

print('\n\nCUDA available: {}\n'.format(torch.cuda.is_available()))

def main(
    args = None
    ):

    parser = argparse.ArgumentParser(description = 'Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help = 'Dataset type, must be csv.')

    parser.add_argument('--csv_train', help = 'Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help = 'Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help = 'Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help = 'Resnet depth, must be one of 18, 34, 50, 101, 152', type = int, default = 50)
    parser.add_argument('--epochs', help = 'Number of epochs', type = int, default = 100)

    parser = parser.parse_args(args)

    # Default parameters
    amplitude = 4095.
    resizer_min_size = 512 #608
    resizer_max_size = 1024

    batch_size = 8

    if not os.path.exists("retinanet_models"):
        os.mkdir("retinanet_models")

    # Create the data loaders
    if parser.dataset == 'csv':
        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on csv,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on csv,')

        transform = transforms.Compose([Normalizer(amplitude = amplitude), Resizer(min_side = resizer_min_size, max_side = resizer_max_size)])
        dataset_train = CSVDataset(train_file = parser.csv_train, class_list = parser.csv_classes, transform = transform)

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            transform = transforms.Compose([Normalizer(amplitude = amplitude), Resizer(min_side = resizer_min_size, max_side = resizer_max_size)])
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list = parser.csv_classes, transform = transform)

    else:
        raise ValueError('Dataset type not understood (must be csv), exiting.')


    sampler = AspectRatioBasedSampler(dataset_train, batch_size = batch_size, drop_last = False)
    dataloader_train = DataLoader(dataset_train, num_workers = 4, collate_fn = collater, batch_sampler = sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size = 1, drop_last = False)
        dataloader_val = DataLoader(dataset_val, num_workers = 4, collate_fn = collater, batch_sampler = sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes = dataset_train.num_classes(), pretrained = False)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes = dataset_train.num_classes(), pretrained = False)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes = dataset_train.num_classes(), pretrained = False)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes = dataset_train.num_classes(), pretrained = False)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes = dataset_train.num_classes(), pretrained = False)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    # DataParallel is not safe.
    # nn.parallel.DistributedParallel() is recommended.
    # if torch.cuda.is_available():
    #     retinanet = torch.nn.DataParallel(retinanet).cuda()
    # else:
    #     retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr = 5e-4) #lr = 1e-5
    # regularization_l2 = 0.0001
    # optimizer = optim.SGD(retinanet.parameters(), lr=5e-4, weight_decay=regularization_l2, momentum=0.9)


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience = 10, verbose=True)

    loss_hist = collections.deque(maxlen = 500)

    retinanet.train()
    # retinanet.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.training = True
        # retinanet.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda()])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                # classification_loss = classification_loss.mean()
                # regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss.item()))

                epoch_loss.append(float(loss.item()))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue
        # evaluation mode
        retinanet.eval()
        retinanet.training = False
        # retinanet.module.freeze_bn()
        if parser.dataset == 'csv' and parser.csv_val is not None:
            print('Evaluating dataset')
            # with torch.no_grad():
            mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.state_dict(), os.path.join("retinanet_models", '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num)))
        # training mode
        # retinanet.train()
        # retinanet.training = True

    retinanet.eval()

    torch.save(retinanet.state_dict(), os.path.join('retinanet_models', 'model_final.pt'))

    return



if __name__ == "__main__":
    main()
