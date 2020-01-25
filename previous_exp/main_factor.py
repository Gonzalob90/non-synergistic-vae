import numpy as np
from collections import OrderedDict, Counter

import torch
import visdom
import torch.nn.functional as F
import torch.optim as optim
import os

from ops import recon_loss, kl_div, permute_dims, kl_div_uni_dim
from utils import traverse, DataGather
from model import VAE, Discriminator
from test_plot_gt import plot_gt_shapes

from test import greedy_policy_Smax_discount_worst

class Trainer_Factor():

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

        self.alpha = args.alpha
        self.omega = args.omega

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

        self.nets = [self.VAE, self.D]

        # Visdom
        self.viz_on = args.viz_port
        self.win_id = dict(recon="win_recon", kl="win_kl", syn='win_syn',
                           tc_loss="win_tc", disc="win_disc",
                           l0="win_l0",
                           l1="win_l1",
                           l2="win_l2",
                           l3="win_l3",
                           l4="win_l4",
                           l5="win_l5",
                           l6="win_l6",
                           l7="win_l7",
                           l8="win_l8",
                           l9="win_l9"
                           )
        self.line_gather = DataGather('iter', 'recon', 'kl', 'syn', 'tc_loss', 'disc',
                                      'l0', 'l1', 'l2', 'l3', 'l4', 'l5',
                                      'l6', 'l7', 'l8', 'l9')

        if self.viz_on:
            self.viz_port = args.viz_port
            self.viz = visdom.Visdom(port=self.viz_port, use_incoming_socket=False)
            self.viz_il_iter = args.viz_il_iter
            self.viz_la_iter = args.viz_la_iter

            if not self.viz.win_exists(win=self.win_id["kl"]):
                self.viz_init()


    def train(self):

        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        epochs = int(np.ceil(self.steps)/len(self.dataloader))
        print("number of epochs {}".format(epochs))

        step = 0
        c = Counter()
        d = Counter()

        for e in range(epochs):
        #for e in range(1):

            for x_true1, x_true2 in self.dataloader:

                #if step == 50: break

                step += 1

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

                # Discriminator
                x_true2 = x_true2.unsqueeze(1).to(self.device)
                z_prime = self.VAE(x_true2, decode = False)[3]
                z_perm = permute_dims(z_prime).detach() ## detaches the output from the graph. no gradient will be backproped along this variable.
                D_z_perm = self.D(z_perm)

                # Discriminator loss
                d_loss = 0.5 * (F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_perm, ones))
                #print("d_loss {}".format(d_loss))


                # Optimise Discriminator
                self.optim_D.zero_grad()
                d_loss.backward()
                self.optim_D.step()


                if self.args.sample == "sample":
                    # Discriminator
                    x_true2 = x_true2.unsqueeze(1).to(self.device)
                    parameters = self.VAE(x_true2, decode=False)
                    mu_prime = parameters[1]
                    logvar_prime = parameters[2]

                elif self.args.sample == "no_sample":
                    mu_prime = mu
                    logvar_prime = logvar

                ##################
                # Synergy Max

                # Step 1: compute the argmax of D kl (q(ai | x(i)) || )
                if self.args.policy == "greedy":
                    best_ai, worst_ai = greedy_policy_Smax_discount_worst(self.z_dim, mu_prime, logvar_prime,
                                                                          alpha=self.omega)
                    # print("step {}, latents best {}".format(step, best_ai))
                    # print("step {}, latents worst{}".format(step, worst_ai))

                    c.update(best_ai)
                    d.update(worst_ai)

                # if self.args.policy == "e-greedy":
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
                    # print("I max {}".format(I_max))
                    # I_max_dim = kl_div_uni_dim(mu_syn, logvar_syn).sum(1).mean()
                    # print("I max dim {}".format(I_max_dim))

                # Step 3: Use it in the loss

                # IN THIS CASE ALPHA SHOULD BE GREATER THAN 0, SOMETHING LIKE 2-4.
                syn_loss = self.alpha * I_max.detach()

                # Logging
                if step % self.args.log_interval == 0:

                    O = OrderedDict(
                        [(i, str(round(count / sum(c.values()) * 100.0, 3)) + '%') for i, count in c.most_common()])

                    P = OrderedDict(
                        [(i, str(round(count / sum(d.values()) * 100.0, 3)) + '%') for i, count in d.most_common()])

                    print("Step {}".format(step))
                    print("Recons. Loss = " + "{:.4f}".format(vae_recon_loss))
                    print("KL Loss = " + "{:.4f}".format(vae_kl))
                    print("TC Loss = " + "{:.4f}".format(tc_loss))
                    print("Factor VAE Loss = " + "{:.4f}".format(vae_loss))
                    print("D loss = " + "{:.4f}".format(d_loss))

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
                    filename = 'traversal_' + str(step) + '.png'
                    filepath = os.path.join(self.args.output_dir, filename)
                    traverse(self.net_mode, self.VAE, self.test_imgs, filepath)

                # Saving plot gt vs predicted
                if not step % self.args.gt_interval:
                    filename = 'gt_' + str(step) + '.png'
                    filepath = os.path.join(self.args.output_dir, filename)
                    plot_gt_shapes(self.net_mode, self.VAE, self.dataloader_gt, filepath)


                # Gather data
                if self.viz_on and (step % self.viz_il_iter == 0):

                    Q = OrderedDict(
                        [(i, round(count / sum(c.values()) * 100.0, 3)) for i, count in c.items()])

                    H = dict()
                    for k in range(10):
                        if k in Q:
                            H[k] = Q[k]
                        else:
                            H[k] = 0.0

                    self.line_gather.insert(iter=step,
                                            recon=vae_recon_loss.item(),
                                            kl=vae_kl.item(),
                                            tc_loss=tc_loss.item(),
                                            disc=d_loss.item(),
                                            syn=syn_loss.item(),
                                            l0 = H[0],
                                            l1 = H[1],
                                            l2 = H[2],
                                            l3 = H[3],
                                            l4 = H[4],
                                            l5 = H[5],
                                            l6 = H[6],
                                            l7 = H[7],
                                            l8 = H[8],
                                            l9 = H[9]
                                            )

                # Visualise data
                if self.viz_on and (step % self.viz_la_iter == 0):
                    self.visualize_line()
                    self.line_gather.flush()


    def visualize_line(self):

        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        recon = torch.Tensor(data['recon'])
        kld = torch.Tensor(data['kl'])
        syn = torch.Tensor(data['syn'])
        tc_loss = torch.Tensor(data['tc_loss'])
        disc = torch.Tensor(data['disc'])

        #print(syn)
        count0 = torch.Tensor(data['l0'])
        #print(count0)
        count1 = torch.Tensor(data['l1'])
        count2 = torch.Tensor(data['l2'])
        count3 = torch.Tensor(data['l3'])
        count4 = torch.Tensor(data['l4'])
        count5 = torch.Tensor(data['l5'])
        count6 = torch.Tensor(data['l6'])
        count7 = torch.Tensor(data['l7'])
        count8 = torch.Tensor(data['l8'])
        count9 = torch.Tensor(data['l9'])

        self.viz.line(X=iters,
                      Y=recon,
                      win=self.win_id['recon'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='reconstruction loss'))

        self.viz.line(X=iters,
                      Y=kld,
                      win=self.win_id['kl'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence'))

        self.viz.line(X=iters,
                      Y=syn,
                      win=self.win_id['syn'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='Syn loss'))

        self.viz.line(X=iters,
                      Y=tc_loss,
                      win=self.win_id['tc_loss'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='TC loss'))

        self.viz.line(X=iters,
                      Y=disc,
                      win=self.win_id['disc'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='Discriminator'))

        # Latent 0
        self.viz.line(X=iters,
                      Y=count0,
                      env='/latents',
                      win=self.win_id['l0'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title = 'Latent 0'))

        # Latent 1
        self.viz.line(X=iters,
                      Y=count1,
                      env='/latents',
                      win=self.win_id['l1'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title = 'Latent 1'))

        # Latent 2
        self.viz.line(X=iters,
                      Y=count2,
                      env='/latents',
                      win=self.win_id['l2'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title = 'Latent 2'))

        # Latent 3
        self.viz.line(X=iters,
                      Y=count3,
                      env='/latents',
                      win=self.win_id['l3'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title = 'Latent 3'))

        # Latent 4
        self.viz.line(X=iters,
                      Y=count4,
                      env='/latents',
                      win=self.win_id['l4'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title = 'Latent 4'))

        # Latent 5
        self.viz.line(X=iters,
                      Y=count5,
                      env='/latents',
                      win=self.win_id['l5'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title = 'Latent 5'))

        # Latent 6
        self.viz.line(X=iters,
                      Y=count6,
                      env='/latents',
                      win=self.win_id['l6'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title = 'Latent 6'))

        # Latent 7
        self.viz.line(X=iters,
                      Y=count7,
                      env='/latents',
                      win=self.win_id['l7'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title = 'Latent 7'))

        # Latent 8
        self.viz.line(X=iters,
                      Y=count8,
                      env='/latents',
                      win=self.win_id['l8'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title = 'Latent 8'))

        # Latent 9
        self.viz.line(X=iters,
                      Y=count9,
                      env='/latents',
                      win=self.win_id['l9'],
                      update='append',
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title = "Latent 9"))


    def viz_init(self):
        zero_init = torch.zeros([1])

        self.viz.line(X=zero_init,
                      Y=zero_init,
                      win=self.win_id['recon'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='reconstruction loss',
                        title='Reconstruction loss'))

        self.viz.line(X=zero_init,
                      Y=zero_init,
                      win=self.win_id['kl'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence',
                        title='KL Loss'))


        self.viz.line(X=zero_init,
                      Y=zero_init,
                      win=self.win_id['syn'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='Syn loss',
                          title='Synergy Loss'))

        self.viz.line(X=zero_init,
                      Y=zero_init,
                      win=self.win_id['tc_loss'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='TC loss',
                          title='TC Loss'))

        self.viz.line(X=zero_init,
                      Y=zero_init,
                      win=self.win_id['disc'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='Discrim.',
                          title='Discriminator'))

        # Latent 0
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env='/latents',
                      win=self.win_id['l0'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title='Latent 0'))

        # Latent 1
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env='/latents',
                      win=self.win_id['l1'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title='Latent 1'))

        # Latent 2
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env='/latents',
                      win=self.win_id['l2'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title='Latent 2'))

        # Latent 3
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env='/latents',
                      win=self.win_id['l3'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title='Latent 3'))

        # Latent 4
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env='/latents',
                      win=self.win_id['l4'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title='Latent 4'))

        # Latent 5
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env='/latents',
                      win=self.win_id['l5'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title='Latent 5'))

        # Latent 6
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env='/latents',
                      win=self.win_id['l6'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title='Latent 6'))

        # Lines 7
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env='/latents',
                      win=self.win_id['l7'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title='Latent 7'))

        # Lines 8
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env='/latents',
                      win=self.win_id['l8'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title='Latent 8'))

        # Lines 9
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env='/latents',
                      win=self.win_id['l9'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='% Counts',
                          title='Latent 9'))


    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()