import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim
import os

from ops import recon_loss, kl_div, permute_dims, kl_div_uni_dim
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

        self.alpha = args.alpha
        self.omega = args.omega

        self.nets = [self.VAE, self.D]

    def train(self):

        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        epochs = int(np.ceil(self.steps)/len(self.dataloader))
        print("number of epochs {}".format(epochs))

        step = 0
        # dict of init opt weights
        #dict_init = {a: defaultdict(list) for a in range(10)}
        # dict of VAE opt weights
        #dict_VAE = {a:defaultdict(list) for a in range(10)}

        weights_names = ['encoder.2.weight', 'encoder.10.weight', 'decoder.0.weight', 'decoder.7.weight', 'net.4.weight']

        dict_VAE = defaultdict(list)
        dict_weight = {a: [] for a in weights_names}

        for e in range(epochs):
        #for e in range():

            for x_true1, x_true2 in self.dataloader:

                if step == 11: break

                step += 1

                """

                # TRACKING OF GRADS
                print("GRADS")
                for name, params in self.VAE.named_parameters():

                    if name == 'encoder.2.weight':
                        #size : 32,32,4,4
                        print("Grads: Before VAE optim step {}".format(step))
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

                    if name == 'encoder.10.weight':
                        #size : 32,32,4,4
                        #print("Before VAE optim  encoder step {}".format(step))
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

                    if name == 'decoder.0.weight':

                        #print("Before VAE optim  decoder step {}".format(step))
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

                    if name == 'decoder.7.weight':

                        #print("Before VAE optim  decoder step {}".format(step))
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

                for name, params in self.D.named_parameters():

                    if name == 'net.4.weight':

                        #print("Before VAE optim  discrim step {}".format(step))
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

                """

                # VAE
                x_true1 = x_true1.unsqueeze(1).to(self.device)
                #print("x_true1 size {}".format(x_true1.size()))

                x_recon, mu, logvar, z = self.VAE(x_true1)


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

                #print("Weights: Before VAE, step {}".format(step))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0, 0, :, :]))
                #print("net.4.weight {}".format(self.optim_D.param_groups[0]['params'][4][:5, :5]))


                # Optimise VAE
                self.optim_VAE.zero_grad() #zero gradients the buffer, grads


                """
                print("after zero grad step {}".format(step))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0, 0, :, :]))
                #print("encoder.10.weight {}".format(self.optim_VAE.param_groups[0]['params'][10][:, :2]))
                #print("net.4.weight {}".format(self.optim_D.param_groups[0]['params'][4][:5, :5]))

                # check if the VAE is optimizing the encoder and decoder
                for name, params in self.VAE.named_parameters():

                    if name == 'encoder.2.weight':
                        # size : 32,32,4,4
                        if step == 1:
                            print("name {}, params grad {}".format(name, params.grad))
                        #else:
                            #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                    if name == 'encoder.10.weight':
                        # size : 32,32,4,4
                        if step == 1:
                            print("name {}, params grad {}".format(name, params.grad))
                        #else:
                            #print("name {}, params grad {}".format(name, params.grad[:, :2]))

                for name, params in self.D.named_parameters():

                    if name == 'net.4.weight':
                        # size : 32,32,4,4
                        if step == 1:

                            print("name {}, params grad {}".format(name, params.grad))
                        #else:

                            #print("name {}, params grad {}".format(name, params.grad[:5, :5]))

                """

                vae_loss.backward(retain_graph = True) # grad parameters are populated

                """
                print()
                print("after backward step {}".format(step))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0, 0, :, :]))
                #print("encoder.10.weight {}".format(self.optim_VAE.param_groups[0]['params'][10][:, :2]))
                #print("net.4.weight {}".format(self.optim_D.param_groups[0]['params'][4][:5, :5]))
                # check if the VAE is optimizing the encoder and decoder
                for name, params in self.VAE.named_parameters():
                    if name == 'encoder.2.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                    if name == 'encoder.10.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[:, :2]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                for name, params in self.D.named_parameters():
                    if name == 'net.4.weight':
                        # size : 1000,1000
                        #print("name {}, params grad {}".format(name, params.grad[:5, :5]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))"""


                self.optim_VAE.step() #Does the step
                #print()
                #print("after VAE update step {}".format(step))
                #print("encoder.2.weight size {}".format(self.optim_VAE.param_groups[0]['params'][2].size()))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0,0,:,:]))
                #print("encoder.10.weight {}".format(self.optim_VAE.param_groups[0]['params'][10][:, :2]))
                #print("net.4.weight {}".format(self.optim_D.param_groups[0]['params'][4][:5, :5]))

                """

                # check if the VAE is optimizing the encoder and decoder
                for name, params in self.VAE.named_parameters():
                    if name == 'encoder.2.weight':
                        #size : 32,32,4,4
                        print("After VAE optim step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))


                    if name == 'encoder.10.weight':
                        #size : 20, 128
                        #print("size of {}: {}".format(name, params.grad.size()))
                        #print("After VAE optim  encoder step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[:, :2]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))


                    if name == 'decoder.0.weight':
                        #128,10
                        #print("After VAE optim  decoder linear step {}".format(step))
                        #print("size of {}: {}".format(name, params.grad.size()))
                        #print("name {}, params grad {}".format(name, params.grad[:3, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))


                    if name == 'decoder.7.weight':
                        #print("After VAE optim  decoder step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[1, 1, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))


                for name, params in self.D.named_parameters():

                    if name == 'net.4.weight':
                        #print("After VAE optim  discriminator step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[:5,:5]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))
                        print()

                """

                #print()
                #print("Before Syn step {}".format(step))
                #print("encoder.2.weight size {}".format(self.optim_VAE.param_groups[0]['params'][2].size()))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0, 0, :, :]))
                #print("encoder.10.weight {}".format(self.optim_VAE.param_groups[0]['params'][10][:, :2]))
                #print("net.4.weight {}".format(self.optim_D.param_groups[0]['params'][4][:5, :5]))

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

                #I_max1 = I_max_batch(best_ai, mu, logvar)
                #print("I_max step{}".format(I_max, step))

                # Step 3: Use it in the loss
                syn_loss = self.alpha * I_max
                #print("syn_loss step {}".format(syn_loss, step))



                # Step 4: Optimise Syn term
                self.optim_VAE.zero_grad() # set zeros all the gradients of VAE network

                """
                #print()
                print("after zeros Syn step {}".format(step))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0, 0, :, :]))
                #print("encoder.10.weight {}".format(self.optim_VAE.param_groups[0]['params'][10][:, :2]))
                #print("net.4.weight {}".format(self.optim_D.param_groups[0]['params'][4][:5, :5]))

                for name, params in self.VAE.named_parameters():
                    if name == 'encoder.2.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                    if name == 'encoder.10.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[:, :2]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                    if name == 'decoder.0.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[:, :2]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                    if name == 'decoder.7.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[:, :2]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))


                for name, params in self.D.named_parameters():
                    if name == 'net.4.weight':
                        # size : 1000,1000
                        #print("name {}, params grad {}".format(name, params.grad[:5, :5]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                """

                syn_loss.backward(retain_graph = True) #backprop the gradients

                """

                print()
                print("after Syn backward step {}".format(step))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0, 0, :, :]))
                #print("encoder.10.weight {}".format(self.optim_VAE.param_groups[0]['params'][10][:, :2]))
                #print("net.4.weight {}".format(self.optim_D.param_groups[0]['params'][4][:5, :5]))

                for name, params in self.VAE.named_parameters():
                    if name == 'encoder.2.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                    if name == 'encoder.10.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[:, :2]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                    if name == 'decoder.0.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[:, :2]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                    if name == 'decoder.7.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[:, :2]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))



                for name, params in self.D.named_parameters():
                    if name == 'net.4.weight':
                        # size : 1000,1000
                        #print("name {}, params grad {}".format(name, params.grad[:5, :5]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                """

                self.optim_VAE.step() #Does the update in VAE network parameters

                #print()
                #print("after Syn update step {}".format(step))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0, 0, :, :]))
                #print("encoder.10.weight {}".format(self.optim_VAE.param_groups[0]['params'][10][:, :2]))
                #print("net.4.weight {}".format(self.optim_D.param_groups[0]['params'][4][:5, :5]))


                ###################

                """
                # check if the VAE is optimizing the encoder and decoder
                for name, params in self.VAE.named_parameters():
                    if name == 'encoder.2.weight':
                        # size : 32,32,4,4
                        print("After Syn optim step {}".format(step))
                        # print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))


                    if name == 'encoder.10.weight':
                        # size :
                        #print("After Syn optim  encoder step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[:, :]))
                        #print("name {}, params grad {}".format(name, params.grad[:, :2]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))

                            dim_changes = []
                            for dim in range(20):
                                if np.array_equal(dict_VAE[name][dim, :2], params.grad.numpy()[dim, :2]) == False:
                                    dim_changes.append(dim)
                            print("Changes in dimensions: {}".format(dim_changes))

                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))


                    if name == 'decoder.0.weight':
                        # 1024, 128
                        #print("After Syn optim  decoder linear step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[:5, :2]))
                        #print("name {}, params grad {}".format(name, params.grad[:3, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))


                    if name == 'decoder.7.weight':
                        #print("After Syn optim  decoder step {}".format(step))
                        # print("name {}, params grad {}".format(name, params.grad[1, 1, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))


                for name, params in self.D.named_parameters():

                    if name == 'net.4.weight':
                        #print("After Syn optim  discriminator step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[:5,:5]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))
                        print()
                """
                # Discriminator
                x_true2 = x_true2.unsqueeze(1).to(self.device)
                z_prime = self.VAE(x_true2, decode = False)[3]
                z_perm = permute_dims(z_prime).detach() ## detaches the output from the graph. no gradient will be backproped along this variable.
                D_z_perm = self.D(z_perm)

                # Discriminator loss
                d_loss = 0.5 * (F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_perm, ones))
                #print("d_loss {}".format(d_loss))

                #print("dict VAE {}".format(dict_VAE['encoder.2.weight'][0, 0, :, :]))

                #print("before Disc, step {}".format(step))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0, 0, :, :]))
                #print("net.4.weight {}".format(self.optim_D.param_groups[0]['params'][4][:5, :5]))


                # Optimise Discriminator
                self.optim_D.zero_grad()

                """

                print("after zero grad Disc step {}".format(step))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0, 0, :, :]))
                #print("net.4.weight {}".format(self.optim_D.param_groups[0]['params'][4][:5, :5]))

                for name, params in self.VAE.named_parameters():
                    if name == 'encoder.2.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                    if name == 'encoder.10.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                    if name == 'decoder.0.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                    if name == 'decoder.7.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                for name, params in self.D.named_parameters():
                    if name == 'net.4.weight':
                        # size : 1000,1000
                        #print("name {}, params grad {}".format(name, params.grad[:5, :5]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                """

                d_loss.backward()

                """
                print()
                print("after backward Disc step {}".format(step))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0, 0, :, :]))
                #print("net.4.weight {}".format(self.optim_D.param_groups[0]['params'][4][:5, :5]))
                # check if the VAE is optimizing the encoder and decoder
                for name, params in self.VAE.named_parameters():
                    if name == 'encoder.2.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                    if name == 'encoder.10.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                    if name == 'decoder.0.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                    if name == 'decoder.7.weight':
                        # size : 32,32,4,4
                        #print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))

                for name, params in self.D.named_parameters():
                    if name == 'net.4.weight':
                        # size : 1000,1000
                        #print("name {}, params grad {}".format(name, params.grad[:5, :5]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))
                """

                self.optim_D.step()

                #print("dict VAE {}".format(dict_VAE['encoder.2.weight'][0, 0, :, :]))
                """
                print()
                print("after update disc step {}".format(step))
                #print("encoder.2.weight size {}".format(self.optim_VAE.param_groups[0]['params'][2].size()))
                #print("encoder.2.weight {}".format(self.optim_VAE.param_groups[0]['params'][2][0, 0, :, :]))
                #print("net.4.weight {}".format(self.optim_D.param_groups[0]['params'][4][:5, :5]))

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

                    if name == 'encoder.10.weight':
                        # size :
                        #print("After Syn optim  encoder step {}".format(step))
                        # print("name {}, params grad {}".format(name, params.grad[0, 0, :, :]))
                        #print("name {}, params grad {}".format(name, params.grad[:, :2]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                            # if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()


                        else:
                            print("No change in gradients {}".format(name))


                    if name == 'decoder.0.weight':
                        #1024, 128
                        #print("After Discriminator optim  decoder linear step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[:5, :2]))
                        #print("name {}, params grad {}".format(name, params.grad[:3, :]))


                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))


                    if name == 'decoder.7.weight':
                        #print("After Discriminator optim  decoder step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[1, 1, :, :]))
                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))


                for name, params in self.D.named_parameters():

                    if name == 'net.4.weight':
                        #print("After Discriminator optim  decoder step {}".format(step))
                        #print("name {}, params grad {}".format(name, params.grad[:5, :5]))

                        if np.array_equal(dict_VAE[name], params.grad.numpy()) == False:
                        #if dict_VAE[name] != tuple(params.grad.numpy()):
                            print("Change in gradients {}".format(name))
                            dict_VAE[name] = params.grad.numpy().copy()
                        else:
                            print("No change in gradients {}".format(name))
                        print()"""

                # Logging
                if step % self.args.log_interval == 0:

                    print("Step {}".format(step))
                    print("Recons. Loss = " + "{:.4f}".format(vae_recon_loss))
                    print("KL Loss = " + "{:.4f}".format(vae_kl))
                    print("TC Loss = " + "{:.4f}".format(tc_loss))
                    print("Factor VAE Loss = " + "{:.4f}".format(vae_loss))
                    print("D loss = " + "{:.4f}".format(d_loss))
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
