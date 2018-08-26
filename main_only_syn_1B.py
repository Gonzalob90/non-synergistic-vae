import numpy as np
from collections import OrderedDict, Counter

import torch
import torch.nn.functional as F
import torch.optim as optim
import os

from ops import recon_loss, kl_div, permute_dims, kl_div_uni_dim
from utils import traverse
from model import VAE

from test import greedy_policy_Smax_discount, I_max_batch, e_greedy_policy_Smax_discount


torch.set_printoptions(precision=6)

class Trainer1B():

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
        c = Counter()

        for e in range(epochs):
        #for e in range():

            for x_true1, x_true2 in self.dataloader:

                #if step == 30: break

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
                if self.args.policy == "greedy":
                    best_ai = greedy_policy_Smax_discount(self.z_dim, mu, logvar,alpha=self.omega)
                    print("step {}, latents {}".format(step, best_ai))
                    c.update(best_ai)

                if self.args.policy == "e-greedy":
                    best_ai = e_greedy_policy_Smax_discount(self.z_dim, mu, logvar, alpha=self.omega, epsilon=self.epsilon)
                    print("step {}, latents {}".format(step, best_ai))
                    c.update(best_ai)

                # Step 2: compute the Imax
                mu_syn = mu[:, best_ai]
                logvar_syn = logvar[:, best_ai]

                if len(mu_syn.size()) == 1:
                    I_max = kl_div_uni_dim(mu_syn, logvar_syn).mean()
                    # print("here")
                else:

                    I_max = kl_div(mu_syn, logvar_syn)
                    #print("I max {}".format(I_max))
                    #I_max_dim = kl_div_uni_dim(mu_syn, logvar_syn).sum(1).mean()
                    #print("I max dim {}".format(I_max_dim))

                # Step 3: Use it in the loss
                syn_loss = self.alpha * I_max

                #print()
                #print("WEIGHTS BEFORE")
                #print("WEIGHTS BEFORE UPDATE STEP {}".format(step))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0, 0, :, :]))
                #print("encoder.10.weight first 5 samples {}".format(self.optim_VAE.param_groups[0]['params'][10][:10, :10]))


                # Step 4: Optimise Syn term
                self.optim_VAE.zero_grad() # set zeros all the gradients of VAE network
                syn_loss.backward() #backprop the gradients

                """
                print("GRADS")
                for name, params in self.VAE.named_parameters():

                    if name == 'encoder.10.weight':
                        # size : 32,32,4,4

                        print()
                        print("params grad {}".format(params.grad[1, :10]))
                        print()
                        print("name {}, params grad {}".format(name, params.grad[:10, :10]))
                """

                self.optim_VAE.step() #Does the update in VAE network parameters

                #print()
                #print("WEIGHTS AFTER UPDATE STEP {}".format(step))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0, 0, :, :]))
                #print("encoder.10.weight first 5 samples {}".format(self.optim_VAE.param_groups[0]['params'][10][:10, :10]))


                # Logging
                if step % self.args.log_interval == 0:

                    O = OrderedDict(
                        [(i, str(round(count / sum(c.values()) * 100.0, 3)) + '%') for i, count in c.most_common()])

                    print("Step {}".format(step))
                    print("Recons. Loss = " + "{:.4f}".format(vae_recon_loss))
                    print("KL Loss = " + "{:.4f}".format(vae_kl))
                    print("VAE Loss = " + "{:.4f}".format(vae_loss))
                    print("best_ai {}".format(best_ai))
                    print("I_max {}".format(I_max))
                    print("Syn loss {:.4f}".format(syn_loss))
                    for k, v in O.items():
                        print("latent {}: {}".format(k, v))

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
