import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import os

from ops import recon_loss, kl_div, permute_dims, kl_div_uni_dim
from utils import traverse
from model import VAE
from test_plot_gt import plot_gt_shapes

torch.set_printoptions(precision=6)

class Trainer_BVAE_gt():

    def __init__(self, args, dataloader, device, test_imgs, dataloader_gt):

        self.device = device
        self.args = args
        self.dataloader = dataloader
        self.dataloader_gt = dataloader_gt

        # Data
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.steps = args.steps
        self.test_imgs = test_imgs

        # Networks & Optimizers
        self.z_dim = args.z_dim
        self.gamma = args.gamma
        self.beta = 4.0

        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE

        self.VAE = VAE(self.z_dim).to(self.device)
        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))

        #self.optim_VAE = optim.SGD(self.VAE.parameters(), lr=1.0)

        self.alpha = args.alpha
        self.omega = args.omega
        self.epsilon = args.epsilon

        self.nets = [self.VAE]

    def train(self):

        self.net_mode(train=True)

        epochs = int(np.ceil(self.steps)/len(self.dataloader))
        print("number of epochs {}".format(epochs))
        print("Run VAE with gt plots and traversals")

        step = 0

        for e in range(epochs):
        #for e in range():

            for x_true1, x_true2 in self.dataloader:

                #if step == 2: break
                step += 1

                # VAE
                x_true1 = x_true1.unsqueeze(1).to(self.device)

                x_recon, mu, logvar, z = self.VAE(x_true1)

                # Reconstruction and KL

                vae_recon_loss = recon_loss(x_true1, x_recon)
                vae_kl = kl_div(mu, logvar)

                # VAE loss
                vae_loss = vae_recon_loss + self.beta * vae_kl

                # Optimise VAE
                self.optim_VAE.zero_grad() #zero gradients the buffer, grads
                vae_loss.backward() # grad parameters are populated
                self.optim_VAE.step() #Does the step

                # Logging
                if step % self.args.log_interval == 0:

                    print("Step {}".format(step))
                    print("Recons. Loss = " + "{:.4f}".format(vae_recon_loss))
                    print("KL Loss = " + "{:.4f}".format(vae_kl))
                    print("VAE Loss = " + "{:.4f}".format(vae_loss))
                    print()

                # Saving traverse
                if not step % self.args.save_interval:
                    filename = 'vae_traversal_' + str(step) + '.png'
                    filepath = os.path.join(self.args.output_dir, filename)
                    traverse(self.net_mode, self.VAE, self.test_imgs, filepath)


                # Saving plot gt vs predicted
                if not step % self.args.gt_interval:
                    filename = 'vae_gt_' + str(step) + '.png'
                    filepath = os.path.join(self.args.output_dir, filename)
                    plot_gt_shapes(self.net_mode, self.VAE, self.dataloader_gt, filepath)



    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()