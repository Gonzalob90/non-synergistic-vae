import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import os

from ops import recon_loss, kl_div, permute_dims, kl_div_uni_dim
from utils import traverse
from model import VAE, Discriminator, Discriminator_syn

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

        self.lr_D = args.lr_D
        self.beta1_D = args.beta1_D
        self.beta2_D = args.beta2_D

        self.lr_D_syn = args.lr_D_syn
        self.beta1_D_syn = args.beta1_D_syn
        self.beta2_D_syn = args.beta2_D_syn


        self.VAE = VAE(self.z_dim).to(self.device)
        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))

        self.D = Discriminator(self.z_dim).to(self.device)
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                  betas=(self.beta1_D, self.beta2_D))

        self.D_syn = Discriminator_syn(self.z_dim).to(self.device)
        self.optim_D_syn = optim.Adam(self.D_syn.parameters(), lr=self.lr_D_syn,
                                  betas=(self.beta1_D_syn, self.beta2_D_syn))

        self.alpha = args.alpha

        self.nets = [self.VAE, self.D, self.D_syn]

    def train(self):

        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        epochs = int(np.ceil(self.steps)/len(self.dataloader))
        print("number of epochs {}".format(epochs))

        step = 0

        for e in range(epochs):
        #for e in range():

            for x_true1, x_true2 in self.dataloader:

                if step == 1: break

                step += 1

                # VAE
                x_true1 = x_true1.unsqueeze(1).to(self.device)
                x_recon, mu, logvar, z = self.VAE(x_true1)


                # Reconstruction and KL
                vae_recon_loss = recon_loss(x_true1, x_recon)
                vae_kl = kl_div(mu, logvar)

                # Total Correlation
                D_z = self.D(z)
                tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

                # Synergy term
                best_ai = self.D_syn(mu,logvar)
                best_ai_labels = torch.bernoulli(best_ai)

                # TODO Copy to an empty tensor

                mu[best_ai_labels == 0] = 0
                logvar_syn[best_ai_labels == 0] = 0

                # TODO For to KL

                for i in range(self.batch_size):
                    mu_syn_s = mu_syn[i][mu_syn[i]!=0]


                if len(mu_syn.size()) == 1:
                    syn_loss = kl_div_uni_dim(mu_syn, logvar_syn).mean()
                    # print("here")
                else:
                    syn_loss = kl_div(mu_syn, logvar_syn)

                # VAE loss
                vae_loss = vae_recon_loss + vae_kl + self.gamma * tc_loss + self.alpha * syn_loss

                # Optimise VAE
                self.optim_VAE.zero_grad() #zero gradients the buffer, grads
                vae_loss.backward(retain_graph = True) # grad parameters are populated
                self.optim_VAE.step() #Does the step


                # TODO Check the best greedy policy
                # Discriminator Syn
                real_seq = greedy_policy_Smax_discount(self.z_dim, mu, logvar, 0.8).detach
                d_syn_loss = recon_loss(real_seq, best_ai)

                # Optimise Discriminator Syn
                self.optim_D_syn.zero_grad()  # set zeros all the gradients of VAE network
                d_syn_loss.backward(retain_graph=True)  # backprop the gradients
                self.optim_D_syn.step()  # Does the update in VAE network parameters

                # Discriminator TC
                x_true2 = x_true2.unsqueeze(1).to(self.device)
                z_prime = self.VAE(x_true2, decode = False)[3]
                z_perm = permute_dims(z_prime).detach() ## detaches the output from the graph. no gradient will be backproped along this variable.
                D_z_perm = self.D(z_perm)

                # Discriminator TC loss
                d_loss = 0.5 * (F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_perm, ones))

                # Optimise Discriminator TC
                self.optim_D.zero_grad()
                d_loss.backward()
                self.optim_D.step()


                # Logging
                if step % self.args.log_interval == 0:

                    print("Step {}".format(step))
                    print("Recons. Loss = " + "{:.4f}".format(vae_recon_loss))
                    print("KL Loss = " + "{:.4f}".format(vae_kl))
                    print("TC Loss = " + "{:.4f}".format(tc_loss))
                    print("Syn Loss = " + "{:.4f}".format(syn_loss))
                    print("Factor VAE Loss = " + "{:.4f}".format(vae_loss))
                    print("D loss = " + "{:.4f}".format(d_loss))
                    print("best_ai {}".format(best_ai))
                    print("Syn loss {:.4f}".format(syn_loss))


                # Saving
                if not step % self.args.save_interval:
                    filename = 'traversal_' + str(step) + '.png'
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
