import os
import argparse
import shutil

import torch
import numpy as np

from main_syn_1A import Trainer1A
from main_only_syn_1B import Trainer1B
from main_only_syn_1C import Trainer1C
from main_only_syn_1D import Trainer1D
from main_only_syn_1B_MLP import Trainer1B_MLP
from main_only_syn_1E import Trainer1E
from main_only_syn_1E_MLP import Trainer1E_MLP
from main_only_syn_1F import Trainer1F
from main_only_syn_1F_MLP import Trainer1F_MLP
from test_1F_graphs import Trainer1F_test
from main_BVAE_MLP import Trainer_BVAE
from main_test_celeba import Trainer1F_celeba
from main_only_syn_1F_plot_gt import  Trainer1F_gt
from main_VAE import Trainer_VAE
from main_VAE_plot_gt import Trainer_VAE_gt
from main_factor import Trainer_Factor
from main_BVAE import Trainer_BVAE_conv


from dataset import get_dsprites_dataloader, get_celeba_dataloader, get_celeba_dataloader_gpu, get_dsprites_dataloader_gt

DATASETS = {'dsprites': [(1, 64, 64), get_dsprites_dataloader],
            'celeba_1': [(3, 64, 64), get_celeba_dataloader],
            'celeba': [(3, 64, 64), get_celeba_dataloader_gpu]}


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

    # Basic hyperparameters
    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the gaussian latents')
    parser.add_argument('--gamma', default=6.4, type=float, help='coefficient of density-ratio term')
    parser.add_argument('--alpha', default=1.5, type=float, help='coefficient of synergy term')
    parser.add_argument('--omega', default=0.8, type=float, help='coefficient of the greedy policy')

    # Synergy
    parser.add_argument('--metric', default='1B', type=str, help="Synergy metrics")
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

    parser.add_argument('--dataset', default='dsprites', type=str, help="dataset to use")
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--nb-test', type=int, default=9, help='number of test samples to visualize the recons of')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=500)
    parser.add_argument('--plot-interval', type=int, default=500)
    parser.add_argument('--save-interval', type=int, default=2000)
    parser.add_argument('--gt-interval', type=int, default=10000)
    parser.add_argument('--mig-interval', type=int, default=50000)
    parser.add_argument('--seq-interval', type=int, default=2000)

    # Visdom
    parser.add_argument('--viz_on', default=True, help='enable visdom visualization')
    parser.add_argument('--viz_port', default=8097, type=int, help='visdom port number')
    parser.add_argument('--viz_il_iter', default=20, type=int, help='visdom line data logging iter')
    parser.add_argument('--viz_la_iter', default=100, type=int, help='visdom line data applying iter')

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

    # data loader gt
    dataloader_gt = get_dsprites_dataloader_gt(args.batch_size)

    # test images to reconstruct during training
    dataset_size = len(dataloader.dataset)
    print(dataset_size)
    indices = np.hstack((0, np.random.choice(range(1,dataset_size),args.nb_test - 1)))
    test_imgs = torch.empty(args.nb_test, *img_dims).to(device)

    for i, img_idx in enumerate(indices):
        test_imgs[i] = torch.tensor(dataloader.dataset[img_idx][0])

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

    if args.metric == "1B_MLP":

        net = Trainer1B_MLP(args, dataloader, device, test_imgs)
        net.train()

    if args.metric == "1E":

        net = Trainer1E(args, dataloader, device, test_imgs)
        net.train()

    if args.metric == "1E_MLP":

        net = Trainer1E_MLP(args, dataloader, device, test_imgs)
        net.train()

    if args.metric == "1F":

        net = Trainer1F(args, dataloader, device, test_imgs)
        net.train()

    if args.metric == "1F_MLP":

        net = Trainer1F_MLP(args, dataloader, device, test_imgs)
        net.train()


    if args.metric == "test":

        net = Trainer1F_test(args, dataloader, device, test_imgs)
        net.train()

    if args.metric == "BVAE":

        net = Trainer_BVAE(args, dataloader, device, test_imgs)
        net.train()

    if args.metric == "Test1":

        net = Trainer1F_celeba(args, dataloader, device, test_imgs)
        net.train()

    if args.metric == "Test_gt":

        net = Trainer1F_gt(args, dataloader, device, test_imgs, dataloader_gt)
        net.train()

    if args.metric == "VAE":

        net = Trainer_VAE(args, dataloader, device, test_imgs, dataloader_gt)
        net.train()

    if args.metric == "VAE_gt":

        net = Trainer_VAE_gt(args, dataloader, device, test_imgs, dataloader_gt)
        net.train()

    if args.metric == "Factor":

        net = Trainer_Factor(args, dataloader, device, test_imgs, dataloader_gt)
        net.train()

    if args.metric == "BVAE_conv":

        net = Trainer_BVAE_conv(args, dataloader, device, test_imgs, dataloader_gt)
        net.train()

if __name__ == "__main__":
    main()