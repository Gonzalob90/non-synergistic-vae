import numpy as np
from collections import OrderedDict, Counter

import torch
import torch.nn.functional as F
import torch.optim as optim
import os

from ops import recon_loss, kl_div, permute_dims, kl_div_uni_dim, recon_loss_faces
from utils import traverse_faces
from model_celeb import VAE_faces

from test import I_max_batch, e_greedy_policy_Smax_discount, greedy_policy_Smax_discount_worst


torch.set_printoptions(precision=6)

class Trainer1F_celeba_VAE():

    def __init__(self, args, dataloader, device, test_imgs):

        self.device = device
        self.args = args
        self.dataloader = dataloader

        # Data
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.steps = args.steps
        self.test_imgs = test_imgs

        # Networks & Optimizers
        self.z_dim = args.z_dim
        self.gamma = args.gamma

        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE


        self.VAE = VAE_faces(self.z_dim).to(self.device)
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

        step = 0
        #c = Counter()
        #d = Counter()

        for e in range(epochs):
        #for e in range():

            for x_true1, x_true2 in self.dataloader:

                step += 1

                # VAE
                #x_true1 = x_true1.unsqueeze(1).to(self.device)
                x_true1 = x_true1.to(self.device)
                #print(x_true1.size())
                #print()

                # x_true1 are between 0 and 1.
                x_recon, mu, logvar, z = self.VAE(x_true1)
                # Reconstruction and KL

                vae_recon_loss = recon_loss(x_true1, x_recon)
                vae_kl = kl_div(mu, logvar)

                # VAE loss
                vae_loss = vae_recon_loss + vae_kl

                # Optimise VAE
                self.optim_VAE.zero_grad() #zero gradients the buffer, grads
                vae_loss.backward() # grad parameters are populated
                self.optim_VAE.step() #Does the step


                # Logging
                if step % self.args.log_interval == 0:

                    #O = OrderedDict(
                    #    [(i, str(round(count / sum(c.values()) * 100.0, 3)) + '%') for i, count in c.most_common()])

                    #P = OrderedDict(
                    #    [(i, str(round(count / sum(d.values()) * 100.0, 3)) + '%') for i, count in d.most_common()])

                    print("Step {}".format(step))
                    print("Recons. Loss = " + "{:.4f}".format(vae_recon_loss))
                    print("KL Loss = " + "{:.4f}".format(vae_kl))
                    print("VAE Loss = " + "{:.4f}".format(vae_loss))
                    #print("best_ai {}".format(best_ai))
                    #print("worst_ai {}".format(worst_ai))
                    #print("I_max {}".format(I_max))
                    #print("Syn loss {:.4f}".format(syn_loss))
                    #print()
                    #for k, v in O.items():
                    #    print("best latent {}: {}".format(k, v))
                    #print()
                    #for k, v in P.items():
                    #    print("worst latent {}: {}".format(k, v))
                    #print()

                # Saving
                if not step % self.args.save_interval:
                    filename = 'alpha_' + str(self.alpha) + '_traversal_' + str(step) + '.png'
                    filepath = os.path.join(self.args.output_dir, filename)
                    traverse_faces(self.net_mode, self.VAE, self.test_imgs, filepath)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()