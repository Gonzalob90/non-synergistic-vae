import os
import argparse
import shutil

import random
import torch
import numpy as np
from main import Trainer
from dataset import get_dsprites_dataloader

DATASETS = {'dsprites': [(1, 64, 64), get_dsprites_dataloader]}


def _get_dataset(data_arg):
    """Checks if the given dataset is available. If yes, returns
       the input dimensions and dataloader."""
    if data_arg not in DATASETS:
        raise ValueError("Dataset not available!")
    return DATASETS[data_arg]

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)

def parse():
    parser = argparse.ArgumentParser(description='Factor-VAE')

    parser.add_argument('--z_dim', default=10, type=int, help='diemension of the gaussian latents')
    parser.add_argument('--gamma', default=6.4, type=float, help='coefficient of density-ratio term')
    parser.add_argument('--lr_VAE', default=1e-4, type=float, help='learning rate for VAE')
    parser.add_argument('--beta1_VAE', default=0.9, type=float, help='beta1 parameter of Adam for VAE')
    parser.add_argument('--beta2_VAE', default=0.999, type= float, help='beta2 parameter of Adam for VAE')
    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate for Discriminator')
    parser.add_argument('--beta1_D', default=0.5, type=float, help='beta1 parameter of Adam for Discriminator')
    parser.add_argument('--beta2_D', default=0.9, type=float, help='beta2 parameter of Adam for Discriminator')

    parser.add_argument('--batch_size', default=64, type=int, help='number of batches')
    parser.add_argument('--steps', default=1e6, type=float, help='steps to train')

    parser.add_argument('--dataset', default='dsprites', type=str, help="dataset to use")
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--nb-test', type=int, default=9, help='number of test samples to visualize the recons of')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--save-interval', type=int, default=50)

    return parser.parse_args()


def main():

    args = parse()

    #device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # data loader
    img_dims, dataloader = _get_dataset(args.dataset)
    dataloader = dataloader(args.batch_size)

    # test images to reconstruct during training
    dataset_size = len(dataloader.dataset)
    indices = np.hstack((0, np.random.choice(range(1,dataset_size),args.nb_test - 1)))
    test_imgs = torch.empty(args.nb_test, *img_dims).to(device)

    for i, img_idx in enumerate(indices):
        test_imgs[i, 0] = torch.tensor(dataloader.dataset[img_idx][0])

    # Create new dir
    if os.path.isdir(args.output_dir): shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    # train
    net = Trainer(args, dataloader, device, test_imgs)
    net.train()


if __name__ == "__main__":
    main()