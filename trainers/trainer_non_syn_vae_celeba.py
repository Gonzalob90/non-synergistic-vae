import numpy as np
from collections import OrderedDict, Counter

import torch
import torch.optim as optim
import os

from utils.ops import kl_div, kl_div_uni_dim, recon_loss_faces
from utils.utils import traverse_faces
from models.model_vae_celeba import VAECelebA
from utils.syn_ops import greedy_policy_s_max_discount_worst
torch.set_printoptions(precision=6)


class TrainerNonSynVAECelebA:

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

        self.VAE = VAECelebA(self.z_dim).to(self.device)
        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))

        self.alpha = args.alpha
        self.omega = args.omega
        self.epsilon = args.epsilon

        self.nets = [self.VAE]

    def train(self):

        self.net_mode(train=True)

        epochs = int(np.ceil(self.steps)/len(self.dataloader))
        print("number of epochs {}".format(epochs))

        step = 0
        c = Counter()
        d = Counter()

        for e in range(epochs):

            for x_true1, x_true2 in self.dataloader:
                step += 1

                # VAE
                x_true1 = x_true1.to(self.device)
                x_recon, mu, log_var, z = self.VAE(x_true1)

                # Reconstruction and KL
                vae_recon_loss = recon_loss_faces(x_true1, x_recon)
                vae_kl = kl_div(mu, log_var)
                vae_loss = vae_recon_loss + vae_kl

                # Optimise VAE
                self.optim_VAE.zero_grad()
                vae_loss.backward(retain_graph=True)
                self.optim_VAE.step()

                if self.args.sample == "sample":
                    x_true2 = x_true2.to(self.device)
                    parameters = self.VAE(x_true2, decode=False)
                    mu_prime = parameters[1]
                    log_var_prime = parameters[2]

                else:
                    mu_prime = mu
                    log_var_prime = log_var

                # Synergy Max

                # Step 1: compute the arg-max of D kl (q(ai | x(i)) || )
                best_ai, worst_ai = greedy_policy_s_max_discount_worst(self.z_dim, mu_prime, log_var_prime,
                                                                       alpha=self.omega)

                c.update(best_ai)
                d.update(worst_ai)

                # Step 2: compute the I-max
                mu_syn = mu_prime[:, worst_ai]
                log_var_syn = log_var_prime[:, worst_ai]

                if len(mu_syn.size()) == 1:
                    i_max = kl_div_uni_dim(mu_syn, log_var_syn).mean()
                else:
                    i_max = kl_div(mu_syn, log_var_syn)

                # Step 3: Use it in the loss
                syn_loss = self.alpha * i_max  # alpha>0 ~2-4

                # Step 4: Optimise Syn term
                self.optim_VAE.zero_grad()
                syn_loss.backward()  # back-propagate the gradients
                self.optim_VAE.step()  # does the update in VAE network parameters

                # Logging
                if step % self.args.log_interval == 0:

                    O = OrderedDict(
                        [(i, str(round(count / sum(c.values()) * 100.0, 3)) + '%') for i, count in c.most_common()])

                    P = OrderedDict(
                        [(i, str(round(count / sum(d.values()) * 100.0, 3)) + '%') for i, count in d.most_common()])

                    print("Step {}".format(step))
                    print("Recons. Loss = " + "{:.4f}".format(vae_recon_loss))
                    print("KL Loss = " + "{:.4f}".format(vae_kl))
                    print("VAE Loss = " + "{:.4f}".format(vae_loss))
                    print("best_ai {}".format(best_ai))
                    print("worst_ai {}".format(worst_ai))
                    print("I_max {}".format(i_max))
                    print("Syn loss {:.4f}".format(syn_loss))
                    print()
                    for k, v in O.items():
                        print("best latent {}: {}".format(k, v))
                    print()
                    for k, v in P.items():
                        print("worst latent {}: {}".format(k, v))
                    print()

                # Saving
                if not step % self.args.save_interval:
                    filename = self.dataset + '_alpha_' + str(self.alpha) + '_traversal_' + str(step) + '.png'
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
