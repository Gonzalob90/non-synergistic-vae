import os
import argparse
import shutil

import torch
import numpy as np

from trainers.train_non_syn_vae import TrainerNonSynVAE
from trainers.train_non_syn_vae_plots import TrainerNonSynVAEPlots
from trainers.trainer_non_syn_vae_celeba import TrainerNonSynVAECelebA
from dataset import get_dsprites_dataloader, get_celeba_dataloader, get_celeba_dataloader_gpu, \
    get_dsprites_dataloader_gt

DATASETS = {'dsprites': [(1, 64, 64), get_dsprites_dataloader],
            'celeba_1': [(3, 64, 64), get_celeba_dataloader],
            'celeba': [(3, 64, 64), get_celeba_dataloader_gpu]}


def _get_dataset(data_arg):
    """Checks if the given dataset is available. If yes, returns
       the input dimensions and dataloader."""
    if data_arg not in DATASETS:
        raise ValueError("Dataset not available!")
    return DATASETS[data_arg]


def parse():
    parser = argparse.ArgumentParser(description='Factor-VAE')

    # Basic hyperparameters
    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the gaussian latents')
    parser.add_argument('--gamma', default=6.4, type=float, help='coefficient of density-ratio term')
    parser.add_argument('--alpha', default=1.5, type=float, help='coefficient of synergy term')
    parser.add_argument('--omega', default=0.8, type=float, help='coefficient of the greedy policy')

    # Synergy
    parser.add_argument('--policy', default='greedy', type=str, help="policy to use for the Synergy metric")
    parser.add_argument('--epsilon', default=0.05, type=float, help='exploration trade-off for e-greedy policy')
    parser.add_argument('--sample', default='no_sample', type=str, help="sample new values of logvar and mu for the Syn term")

    # Annealing
    parser.add_argument('--warmup_iter', default=7000, type=int, help='annealing value')

    # Optimizers
    parser.add_argument('--lr_VAE', default=1e-4, type=float, help='learning rate for VAE')
    parser.add_argument('--beta1_VAE', default=0.9, type=float, help='beta1 parameter of Adam for VAE')
    parser.add_argument('--beta2_VAE', default=0.999, type= float, help='beta2 parameter of Adam for VAE')

    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate for Discriminator')
    parser.add_argument('--beta1_D', default=0.5, type=float, help='beta1 parameter of Adam for Discriminator')
    parser.add_argument('--beta2_D', default=0.9, type=float, help='beta2 parameter of Adam for Discriminator')

    parser.add_argument('--batch_size', default=64, type=int, help='number of batches')
    parser.add_argument('--steps', default=3e5, type=float, help='steps to train')

    # Miscellaneous
    parser.add_argument('--dataset', default='dsprites', type=str, help="dataset to use")
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--nb-test', type=int, default=9, help='number of test samples to visualize the recons of')
    parser.add_argument('--seed', type=int, default=1)

    # Intervals
    parser.add_argument('--log-interval', type=int, default=500)
    parser.add_argument('--plot-interval', type=int, default=500)
    parser.add_argument('--save-interval', type=int, default=2000)
    parser.add_argument('--gt-interval', type=int, default=10000)
    parser.add_argument('--mig-interval', type=int, default=50000)
    parser.add_argument('--seq-interval', type=int, default=2000)

    # Visdom
    parser.add_argument('--viz_on', default=False, help='enable visdom visualization')
    parser.add_argument('--viz_port', default=8097, type=int, help='visdom port number')
    parser.add_argument('--viz_il_iter', default=20, type=int, help='visdom line data logging iter')
    parser.add_argument('--viz_la_iter', default=100, type=int, help='visdom line data applying iter')

    return parser.parse_args()


def main():

    init_seed = 120
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse()

    #device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using {device}')

    # data loader
    print(f'Using dataset: {args.dataset}')
    img_dims, dataloader = _get_dataset(args.dataset)
    dataloader = dataloader(args.batch_size)

    # data loader gt
    dataloader_gt = get_dsprites_dataloader_gt(args.batch_size)

    # test images to reconstruct during training
    dataset_size = len(dataloader.dataset)
    print(f'Number of examples in dataset: {dataset_size}')
    indices = np.hstack((0, np.random.choice(range(1, dataset_size), args.nb_test - 1)))
    test_imgs = torch.empty(args.nb_test, *img_dims).to(device)

    for i, img_idx in enumerate(indices):
        test_imgs[i] = torch.tensor(dataloader.dataset[img_idx][0])

    # Create new dir
    if os.path.isdir(args.output_dir): shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    if args.viz_on:
        net = TrainerNonSynVAEPlots(args, dataloader, device, test_imgs)
        print('Training Non-Syn VAE, monitoring metrics and latent count')
        net.train()
    elif args.dataset == 'celeba':
        net = TrainerNonSynVAECelebA(args, dataloader, device, test_imgs)
        print('Training Non-Syn VAE, plotting traverse plots for dataset: celeba')
        net.train()
    else:
        net = TrainerNonSynVAE(args, dataloader, device, test_imgs, dataloader_gt)
        print('Training Non-Syn VAE, plotting the GT plots and traverse plots; dataset: dsprites')
        net.train()


if __name__ == "__main__":
    main()