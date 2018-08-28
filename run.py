import os
import argparse
import shutil

import torch
import numpy as np

from main_syn_1A import Trainer1A
from main_only_syn_1B import Trainer1B
from main_only_syn_1C import Trainer1C
from main_only_syn_1D import Trainer1D
from test_sample_main_syn_1A import Test

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

    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the gaussian latents')
    parser.add_argument('--gamma', default=6.4, type=float, help='coefficient of density-ratio term')
    parser.add_argument('--alpha', default=1.5, type=float, help='coefficient of synergy term')
    parser.add_argument('--omega', default=0.8, type=float, help='coefficient of the greedy policy')

    parser.add_argument('--metric', default='1B', type=str, help="Synergy metrics")
    parser.add_argument('--policy', default='greedy', type=str, help="policy to use for the Synergy metric")
    parser.add_argument('--epsilon', default=0.05, type=float, help='exploration trade-off for e-greedy policy')
    parser.add_argument('--sample', default='no_sample', type=str, help="sample new values of logvar and mu for the Syn term")

    parser.add_argument('--lr_VAE', default=1e-4, type=float, help='learning rate for VAE')
    parser.add_argument('--beta1_VAE', default=0.9, type=float, help='beta1 parameter of Adam for VAE')
    parser.add_argument('--beta2_VAE', default=0.999, type= float, help='beta2 parameter of Adam for VAE')

    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate for Discriminator')
    parser.add_argument('--beta1_D', default=0.5, type=float, help='beta1 parameter of Adam for Discriminator')
    parser.add_argument('--beta2_D', default=0.9, type=float, help='beta2 parameter of Adam for Discriminator')

    parser.add_argument('--batch_size', default=64, type=int, help='number of batches')
    parser.add_argument('--steps', default=3e5, type=float, help='steps to train')

    parser.add_argument('--dataset', default='dsprites', type=str, help="dataset to use")
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--nb-test', type=int, default=9, help='number of test samples to visualize the recons of')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=500)
    parser.add_argument('--plot-interval', type=int, default=500)
    parser.add_argument('--save-interval', type=int, default=2000)
    parser.add_argument('--seq-interval', type=int, default=2000)

    return parser.parse_args()


def main():

    args = parse()

    #device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

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
    if args.metric == "1A":

        net = Trainer1A(args, dataloader, device, test_imgs)
        net.train()

    if args.metric == "1B":

        net = Trainer1B(args, dataloader, device, test_imgs)
        net.train()

    if args.metric == "1C":

        net = Trainer1C(args, dataloader, device, test_imgs)
        net.train()

    if args.metric == "1D":

        net = Trainer1D(args, dataloader, device, test_imgs)
        net.train()

    if args.metric == "test":

        net = Test(args, dataloader, device, test_imgs)
        net.train()


if __name__ == "__main__":
    main()