import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim
import os

from ops import recon_loss, kl_div, permute_dims, kl_div_uni_dim
from utils import traverse
from model import VAE

from test import greedy_policy_Smax_discount, I_max_batch


torch.set_printoptions(precision=6)

class Trainer():

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

        self.VAE = VAE(self.z_dim).to(self.device)
        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))

        self.alpha = args.alpha
        self.omega = args.omega

        self.nets = [self.VAE]

    def train(self):

        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        epochs = int(np.ceil(self.steps)/len(self.dataloader))
        print("number of epochs {}".format(epochs))

        step = 0


        weights_names = ['encoder.2.weight', 'encoder.10.weight', 'decoder.0.weight', 'decoder.7.weight', 'net.4.weight']

        dict_VAE = defaultdict(list)
        dict_weight = {a: [] for a in weights_names}

        for e in range(epochs):
        #for e in range():

            for x_true1, x_true2 in self.dataloader:

                #if step == 10: break

                step += 1

                # VAE
                x_true1 = x_true1.unsqueeze(1).to(self.device)

                x_recon, mu, logvar, z = self.VAE(x_true1)

                # Reconstruction and KL
                vae_recon_loss = recon_loss(x_true1, x_recon)
                vae_kl = kl_div(mu, logvar)

                # VAE loss
                vae_loss = vae_recon_loss + vae_kl

                # Optimise VAE
                self.optim_VAE.zero_grad() #zero gradients the buffer, grads
                vae_loss.backward(retain_graph = True) # grad parameters are populated
                self.optim_VAE.step() #Does the step

                ##################
                #Synergy Max

                # Step 1: compute the argmax of D kl (q(ai | x(i)) || )
                best_ai = greedy_policy_Smax_discount(self.z_dim, mu,logvar,alpha=self.omega)

                # Step 2: compute the Imax
                mu_syn = mu[:, best_ai]
                logvar_syn = logvar[:, best_ai]

                if len(mu_syn.size()) == 1:
                    I_max = kl_div_uni_dim(mu_syn, logvar_syn).mean()
                    # print("here")
                else:
                    I_max = kl_div(mu_syn, logvar_syn)


                # Step 3: Use it in the loss
                syn_loss = self.alpha * I_max

                # Step 4: Optimise Syn term
                self.optim_VAE.zero_grad() # set zeros all the gradients of VAE network
                syn_loss.backward() #backprop the gradients
                self.optim_VAE.step() #Does the update in VAE network parameters

                # Logging
                if step % self.args.log_interval == 0:

                    print("Step {}".format(step))
                    print("Recons. Loss = " + "{:.4f}".format(vae_recon_loss))
                    print("KL Loss = " + "{:.4f}".format(vae_kl))
                    print("VAE Loss = " + "{:.4f}".format(vae_loss))
                    print("best_ai {}".format(best_ai))
                    print("I_max {}".format(I_max))
                    print("Syn loss {:.4f}".format(syn_loss))

                # Saving
                if not step % self.args.save_interval:
                    filename = 'alpha_' + str(self.alpha) + '_traversal_' + str(step) + '.png'
                    filepath = os.path.join(self.args.output_dir, filename)
                    traverse(self.net_mode, self.VAE, self.test_imgs, filepath)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()
