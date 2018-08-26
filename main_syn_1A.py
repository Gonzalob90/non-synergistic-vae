import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim
import os

from ops import recon_loss, kl_div, kl_div_uni_dim, kl_div_mean_syn_1A
from utils import traverse
from model import VAE, Discriminator

from test import greedy_policy_Smax_discount, I_max_batch, e_greedy_policy_Smax_discount
from plots_matrices import plot_matrices_1

torch.set_printoptions(precision=6)

class Trainer1A():

    def __init__(self, args, dataloader, device, test_imgs):

        self.device = device
        self.args = args
        self.dataloader = dataloader

        # Data
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.steps = args.steps
        self.test_imgs = test_imgs
        self.seq_interval_1= args.seq_interval - 10

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

        self.alpha = args.alphafd
        self.omega = args.omega
        self.epsilon = args.epsilon

        self.nets = [self.VAE]

    def train(self):

        self.net_mode(train=True)


        epochs = int(np.ceil(self.steps)/len(self.dataloader))
        print("number of epochs {}".format(epochs))

        step = 0

        matrix_list = []

        for e in range(epochs):
        #for e in range(2):

            for x_true1, x_true2 in self.dataloader:

                #if step == 1: break

                step += 1

                # VAE
                x_true1 = x_true1.unsqueeze(1).to(self.device)
                #print("x_true1 size {}".format(x_true1.size()))

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

                syn_flag = False
                S_max = []

                # Start the beginning
                if step % self.args.seq_interval == self.seq_interval_1 + 1:
                    matrix_list = []
                if step % self.args.seq_interval in np.arange(self.seq_interval_1 + 1,self.args.seq_interval + 1):
                    print(step % self.args.seq_interval)
                    syn_flag = True

                for s in range(self.batch_size):

                    # slice one sample
                    mu_s = mu[s, :]
                    logvar_s = logvar[s, :]

                    # Step 1: compute the argmax of D kl (q(ai | x(i)) || )
                    if self.args.policy == "greedy":
                        index = greedy_policy_Smax_discount(self.z_dim, mu_s.view([1,-1]), logvar_s.view([1,-1]), alpha=self.omega)
                        #print("step {}, sample {}, latents {}".format(step, s, index))


                    if self.args.policy == "e-greedy":
                        index = e_greedy_policy_Smax_discount(self.z_dim, mu_s.view([1,-1]), logvar_s.view([1,-1]), alpha=self.omega,
                                                                epsilon=self.epsilon)
                        #print("step {}, sample {}, latents {}".format(step, s, index))

                    # get the argmax of D kl (q(ai | x(i)) || )
                    #index = greedy_policy_Smax_discount(self.z_dim, mu_s.view([1,-1]), logvar_s.view([1,-1]), alpha=self.omega)
                    #print("step {}, sample {}, index {}".format(step, s, index))

                    mask1 = torch.zeros_like(mu_s)
                    mask1[index] = 1

                    if syn_flag:
                        matrix_list.append(mask1.numpy())

                    #print(mask1.size())
                    #print(mask1)

                    # get the dims:
                    mu_syn = mask1 * mu_s
                    #print("mu_syn masked {}".format(mu_syn))
                    #print("mu original {}".format(mu_s))
                    logvar_syn = mask1 * logvar_s


                    #print("sample {}, mu_syn {}".format(s, mu_syn))
                    #print("sample {}, logvar_syn {}".format(s, logvar_syn))

                    mu_syn = mu_syn.view([1,-1])
                    logvar_syn = logvar_syn.view([1, -1])

                    if len(mu_syn.size()) == 1:
                        I_max = kl_div_mean_syn_1A(mu_syn, logvar_syn)
                        #print("here")
                    else:
                        I_max = kl_div_mean_syn_1A(mu_syn, logvar_syn)
                        #print("I_max {}".format(I_max))

                        #I_max_dim = kl_div_uni_dim(mu_syn, logvar_syn)
                        #print("I max dim {}".format(I_max_dim))

                    #print(I_max)
                    #print(I_max.size())
                    S_max.append(I_max)
                    #print("SAMPLE {}".format(s))

                # Saving sequence of synergy latents
                if step % self.args.seq_interval == 0:

                    filename1 = 'alpha_' + str(self.alpha) + '_omega_' + str(self.omega) + '_sequence_' + str(step) + '.png'
                    filepath1 = os.path.join(self.args.output_dir, filename1)
                    matrix_syn = np.vstack(matrix_list)
                    plot_matrices_1(matrix_syn.T, filepath1, step)

                #print(S_max)
                Smax = torch.cat(S_max, dim=0)
                #print(Smax)
                #print(Smax.size())
                #print()
                #print(Smax[:10])
                syn_term = Smax.mean(dim = 0)
                #print()

                syn_loss = self.alpha * syn_term
                #print("syn_loss step {}".format(syn_loss, step))


                #print()
                #print("WEIGHTS BEFORE")
                #print("WEIGHTS AFTER UPDATE STEP {}".format(step))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0, 0, :, :]))
                #print("encoder.10.weight first 5 samples {}".format(self.optim_VAE.param_groups[0]['params'][10][:10, :10]))


                # Step 4: Optimise Syn term
                self.optim_VAE.zero_grad()  # set zeros all the gradients of VAE network
                syn_loss.backward()  # backprop the gradients

                """
                print("GRADS")

                for name, params in self.VAE.named_parameters():
                    #if name == 'encoder.2.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                    if name == 'encoder.10.weight':
                        # size : 32,32,4,4
                        print(Smax[:10])

                        print()
                        print("params grad {}".format(params.grad[1, :10]))
                        print()
                        print("name {}, params grad {}".format(name, params.grad[:10, :10]))
                """

                self.optim_VAE.step()  # Does the update in VAE network parameters

                #print()
                #print("WEIGHTS AFTER UPDATE STEP {}".format(step))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0,0,:,:]))
                #print("encoder.10.weight first 5 samples {}".format(self.optim_VAE.param_groups[0]['params'][10][:10, :10]))

                ###### END Syn


                # Logging
                if step % self.args.log_interval == 0:

                    print("Step {}".format(step))
                    print("Recons. Loss = " + "{:.4f}".format(vae_recon_loss))
                    print("KL Loss = " + "{:.4f}".format(vae_kl))
                    print("Factor VAE Loss = " + "{:.4f}".format(vae_loss))
                    print("I_max {}".format(I_max))
                    print("Syn loss {:.4f}".format(syn_loss))


                # Saving traverse images
                if not step % self.args.save_interval:
                    filename = 'alpha_' + str(self.alpha) + '_omega_' + str(self.omega) + '_traversal_' + str(step) + '.png'
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
