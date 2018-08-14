import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim
import os

from ops import recon_loss, kl_div, permute_dims
from utils import traverse
from model import VAE, Discriminator

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

        self.VAE = VAE(self.z_dim).to(self.device)
        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))

        self.D = Discriminator(self.z_dim).to(self.device)
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                  betas=(self.beta1_D, self.beta2_D))

        self.alpha = 0.8

        self.nets = [self.VAE, self.D]

    def train(self):

        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        epochs = int(np.ceil(self.steps)/len(self.dataloader))
        print("number of epochs {}".format(epochs))

        step = 0
        # dict of init opt weights
        dict_init = {a: defaultdict(list) for a in range(10)}
        # dict of VAE opt weights
        #dict_VAE = {a:defaultdict(list) for a in range(10)}
        dict_VAE = {a: defaultdict(tuple) for a in range(10)}
        # dict of disc opt weights
        dict_disc = {a:defaultdict(list) for a in range(10)}

        #for e in range(epochs):
        for e in range(1):

            for x_true1, x_true2 in self.dataloader:

                if step == 2: break

                step += 1

                for name, params in self.VAE.named_parameters():

                    if name == 'encoder.2.weight':
                        #size : 32,32,4,4
                        print("Before VAE optim  encoder step {}".format(step))
                        #if params.grad != None:
                        if step != 1:
                            if np.array_equal(dict_VAE[name], params.grad.numpy()) == False :
                            #if dict_VAE[name] != tuple(params.grad.numpy()):
                                print("Change in gradients {}".format(name))
                                #dict_init[step][name] = params.grad.numpy()
                                dict_VAE[name] = params.grad.numpy().copy()
                            else:
                                print("No change in gradients {}".format(name))
                            #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                        else:
                            print("name {}, params grad {}".format(name, params.grad))
                            #dict_init[step][name] = None
                            dict_VAE[name] = None
                        print()

                    if name == 'decoder.1.weight':

                        print("Before VAE optim  decoder step {}".format(step))
                        #if params.grad != None:
                        if step != 1:

                            if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            #if dict_VAE[name] != tuple(params.grad.numpy()):
                                print("Change in gradients {}".format(name))
                                dict_VAE[name] = params.grad.numpy().copy()
                            else:
                                print("No change in gradients {}".format(name))
                            #print("name {}, params grad {}".format(name, params.grad[:5, :2]))
                        else:
                            print("name {}, params grad {}".format(name, params.grad))
                            #dict_init[step][name] = None
                            dict_VAE[name] = None
                        print()

                    if name == 'decoder.7.weight':

                        print("Before VAE optim  decoder step {}".format(step))
                        #if params.grad != None:
                        if step != 1:

                            if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            #if dict_VAE[name] != tuple(params.grad.numpy()):
                                print("Change in gradients {}".format(name))
                                dict_VAE[name] = params.grad.numpy().copy()
                            else:
                                print("No change in gradients {}".format(name))
                            #print("name {}, params grad {}".format(name, params.grad[1, 1, :, :]))
                        else:
                            print("name {}, params grad {}".format(name, params.grad))
                            #dict_init[step][name] = None
                            dict_VAE[name] = None
                        print()

                for name, params in self.D.named_parameters():

                    if name == 'net.4.weight':

                        print("Before VAE optim  discrim step {}".format(step))
                        #if params.grad != None:
                        if step != 1:

                            if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            #if dict_VAE[name] != tuple(params.grad.numpy()):
                                print("Change in gradients {}".format(name))
                                dict_VAE[name] = params.grad.numpy().copy()
                            else:
                                print("No change in gradients {}".format(name))
                            #print("name {}, params grad {}".format(name, params.grad[1, 1, :, :]))
                        else:
                            print("name {}, params grad {}".format(name, params.grad))
                            #dict_init[step][name] = None
                            dict_VAE[name] = None
                        print()

                # VAE
                x_true1 = x_true1.unsqueeze(1).to(self.device)
                #print("x_true1 size {}".format(x_true1.size()))

                x_recon, mu, logvar, z = self.VAE(x_true1)

                #print("x_recon size {}".format(x_recon.size()))
                #print("mu size {}".format(mu.size()))
                #print("logvar size {}".format(logvar.size()))
                #print("z size {}".format(z.size()))

                # Reconstruction and KL
                vae_recon_loss = recon_loss(x_true1, x_recon)
                #print("vae recon loss {}".format(vae_recon_loss))
                vae_kl = kl_div(mu, logvar)
                #print("vae kl loss {}".format(vae_kl))

                # Total Correlation
                D_z = self.D(z)
                #print("D_z size {}".format(D_z.size()))
                tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
                #print("tc loss {}".format(tc_loss))

                # VAE loss
                vae_loss = vae_recon_loss + vae_kl + self.gamma * tc_loss
                #print("Total VAE loss {}".format(vae_loss))

                # Optimise VAE
                self.optim_VAE.zero_grad() #zero gradients the buffer
                vae_loss.backward(retain_graph = True)
                self.optim_VAE.step() #Does the step


                # check if the VAE is optimizing the encoder and decoder
                for name, params in self.VAE.named_parameters():
                    if name == 'encoder.2.weight':
                        #size : 32,32,4,4
                        print("After VAE optim  encoder step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))
                        print()

                    if name == 'decoder.1.weight':
                        #1024, 128
                        print("After VAE optim  decoder linear step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[:5, :2]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))
                        print()

                    if name == 'decoder.7.weight':
                        print("After VAE optim  decoder step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[1, 1, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))
                        print()

                for name, params in self.D.named_parameters():

                    if name == 'net.4.weight':
                        print("After VAE optim  discriminator step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[:5,:5]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))
                        print()

                ##################
                #Synergy Max


                # Step 1: compute the argmax of D kl (q(ai | x(i)) || )
                best_ai = greedy_policy_Smax_discount(self.z_dim, mu,logvar,alpha=0.8)
                print("best_ai {}".format(best_ai))


                # Step 2: compute the Imax
                I_max = I_max_batch(best_ai,mu,logvar)
                print("I_max {}".format(I_max))


                # Step 3: Use it in the loss
                syn_loss = self.alpha * I_max
                print("syn_loss {}".format(syn_loss))


                # Step 4: Optimise Syn term
                self.optim_VAE.zero_grad() #Does the update in VAE network
                syn_loss.backward()
                self.optim_VAE.step() #Does the update in VAE network


                ###################

                # Discriminator
                x_true2 = x_true2.unsqueeze(1).to(self.device)
                z_prime = self.VAE(x_true2, decode = False)[3]
                z_perm = permute_dims(z_prime).detach() ## detaches the output from the graph. no gradient will be backproped along this variable.
                D_z_perm = self.D(z_perm)

                # Discriminator loss
                d_loss = 0.5 * (F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_perm, ones))
                #print("d_loss {}".format(d_loss))

                #print("dict VAE {}".format(dict_VAE['encoder.2.weight'][0, 0, :, :]))

                # Optimise Discriminator
                self.optim_D.zero_grad()
                d_loss.backward()
                self.optim_D.step()

                #print("dict VAE {}".format(dict_VAE['encoder.2.weight'][0, 0, :, :]))

                for name, params in self.VAE.named_parameters():
                    if name == 'encoder.2.weight':
                        #size : 32,32,4,4
                        print("After Discriminator optim  encoder step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))
                        #print("dict VAE {}".format(dict_VAE[name][0, 0, :, :]))
                        #if np.isclose(dict_VAE[name], params.grad.numpy(), rtol=1e-05, atol=1e-08, equal_nan=False): "Works"
                        #if np.all(abs(dict_VAE[step][name] - params.grad.numpy())) < 1e-7 == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))
                        #print("dict VAE {}".format(dict_VAE[name][0, 0, :, :]))
                        print()

                    if name == 'decoder.1.weight':
                        #1024, 128
                        print("After Discriminator optim  decoder linear step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[:5, :2]))
                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))
                        print()

                    if name == 'decoder.7.weight':
                        print("After Discriminator optim  decoder step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[1, 1, :, :]))
                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))
                        print()

                for name, params in self.D.named_parameters():

                    if name == 'net.4.weight':
                        print("After Discriminator optim  decoder step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[:5, :5]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))
                        print()

                # Logging
                if step % self.args.log_interval == 0:

                    print("Step {}".format(step))
                    print("Recons. Loss = " + "{:.4f}".format(vae_recon_loss))
                    print("KL Loss = " + "{:.4f}".format(vae_kl))
                    print("TC Loss = " + "{:.4f}".format(tc_loss))
                    print("Factor VAE Loss = " + "{:.4f}".format(vae_loss))
                    print("D loss = " + "{:.4f}".format(d_loss))

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
