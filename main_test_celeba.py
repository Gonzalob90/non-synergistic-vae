import numpy as np
from collections import OrderedDict, Counter

import torch
import torch.nn.functional as F
import torch.optim as optim
import os

from ops import recon_loss, kl_div, permute_dims, kl_div_uni_dim
from utils import traverse_faces
from model_celeb import VAE_faces

from test import I_max_batch, e_greedy_policy_Smax_discount, greedy_policy_Smax_discount_worst


torch.set_printoptions(precision=6)

class Trainer1F_celeba():

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
        c = Counter()
        d = Counter()

        for e in range(epochs):
        #for e in range():

            for x_true1, x_true2 in self.dataloader:

                #if step == 2: break

                step += 1


                # VAE
                #x_true1 = x_true1.unsqueeze(1).to(self.device)
                x_true1 = x_true1.to(self.device)
                print(x_true1.size())
                print()

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

                if self.args.sample == "sample":
                    # Discriminator
                    #x_true2 = x_true2.unsqueeze(1).to(self.device)
                    x_true2 = x_true2.to(self.device)
                    parameters = self.VAE(x_true2, decode=False)
                    mu_prime = parameters[1]
                    logvar_prime = parameters[2]

                elif self.args.sample == "no_sample":
                    mu_prime = mu
                    logvar_prime = logvar

                ##################
                #Synergy Max

                # Step 1: compute the argmax of D kl (q(ai | x(i)) || )
                if self.args.policy == "greedy":
                    best_ai, worst_ai = greedy_policy_Smax_discount_worst(self.z_dim, mu_prime, logvar_prime,alpha=self.omega)
                    #print("step {}, latents best {}".format(step, best_ai))
                    #print("step {}, latents worst{}".format(step, worst_ai))

                    c.update(best_ai)
                    d.update(worst_ai)

                #if self.args.policy == "e-greedy":
                #    best_ai, worst_ai = e_greedy_policy_Smax_discount(self.z_dim, mu_prime, logvar_prime, alpha=self.omega, epsilon=self.epsilon)
                #    #print("step {}, latents {}".format(step, best_ai))
                #    c.update(best_ai)
                #    d.update(worst_ai)

                # Step 2: compute the Imax
                mu_syn = mu_prime[:, worst_ai]
                logvar_syn = logvar_prime[:, worst_ai]

                if len(mu_syn.size()) == 1:
                    I_max = kl_div_uni_dim(mu_syn, logvar_syn).mean()
                    # print("here")
                else:

                    I_max = kl_div(mu_syn, logvar_syn)
                    #print("I max {}".format(I_max))
                    #I_max_dim = kl_div_uni_dim(mu_syn, logvar_syn).sum(1).mean()
                    #print("I max dim {}".format(I_max_dim))

                # Step 3: Use it in the loss

                # IN THIS CASE ALPHA SHOULD BE GREATER THAN 0, SOMETHING LIKE 2-4.
                syn_loss = self.alpha * I_max

                """
                print()
                print("WEIGHTS BEFORE UPDATE STEP {}".format(step))
                print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0, 0, :, :]))
                print("mu weights")
                print("encoder.10.weight first 10 samples {}".format(
                    self.optim_VAE.param_groups[0]['params'][10][:10, :10]))
                print()
                print("logvar weights")
                print("encoder.10.weight first 10 samples {}".format(
                    self.optim_VAE.param_groups[0]['params'][10][10:, :10]))
                """

                # Step 4: Optimise Syn term
                self.optim_VAE.zero_grad() # set zeros all the gradients of VAE network
                syn_loss.backward() #backprop the gradients

                """
                print("GRADS")

                for name, params in self.VAE.named_parameters():
                    # if name == 'encoder.2.weight':
                    # size : 32,32,4,4
                    # print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                    if name == 'encoder.10.weight':
                        # size : 32,32,4,4
                        #print(Smax[:10])

                        print()
                        # print("params grad {}".format(params.grad[1, :10]))
                        print()
                        print("mu gradients name {}".format(name))
                        print(params.grad[:10, :10])
                        print()
                        print("logvar gradients name {}".format(name))
                        print(params.grad[10:, :10])

                """


                self.optim_VAE.step() #Does the update in VAE network parameters

                """
                print()
                print("WEIGHTS AFTER UPDATE STEP {}".format(step))
                print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0,0,:,:]))
                print("mu weights")
                print("encoder.10.weight first 10 samples {}".format(
                    self.optim_VAE.param_groups[0]['params'][10][:10, :10]))
                print()
                print("logvar weights")
                print("encoder.10.weight first 10 samples {}".format(
                    self.optim_VAE.param_groups[0]['params'][10][10:, :10]))
                """


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
                    print("I_max {}".format(I_max))
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