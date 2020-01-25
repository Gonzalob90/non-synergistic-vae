import numpy as np
from collections import OrderedDict, Counter

import visdom
import torch
import torch.optim as optim
import os

from utils.ops import recon_loss, kl_div, kl_div_uni_dim
from utils.utils import traverse, DataGather
from models.model_vae import VAE
from utils.syn_ops import greedy_policy_s_max_discount_worst
torch.set_printoptions(precision=6)


class TrainerNonSynVAEPlots:

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
        self.epsilon = args.epsilon

        self.warmup_iter = args.warmup_iter

        self.nets = [self.VAE]

        # Visdom
        self.viz_on = args.viz_port
        self.win_id = dict(recon="win_recon", kl="win_kl", syn='win_syn',
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
        self.line_gather = DataGather('iter', 'recon', 'kl','syn',
                                      'l0','l1','l2','l3','l4','l5',
                                      'l6','l7','l8','l9')

        if self.viz_on:
            self.viz_port = args.viz_port
            self.viz = visdom.Visdom(port=self.viz_port, use_incoming_socket=False)
            self.viz_il_iter= args.viz_il_iter
            self.viz_la_iter = args.viz_la_iter

            if not self.viz.win_exists(win=self.win_id["kl"]):
                self.viz_init()

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
                x_true1 = x_true1.unsqueeze(1).to(self.device)
                x_recon, mu, log_var, z = self.VAE(x_true1)

                # Reconstruction and KL
                vae_recon_loss = recon_loss(x_true1, x_recon)
                vae_kl = kl_div(mu, log_var)
                vae_loss = vae_recon_loss + vae_kl

                # Optimise VAE
                self.optim_VAE.zero_grad()
                vae_loss.backward(retain_graph=True)  # grad parameters are populated
                self.optim_VAE.step()

                # Sampling
                if self.args.sample == "sample":
                    x_true2 = x_true2.unsqueeze(1).to(self.device)
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
                syn_loss = self.alpha * i_max

                # Step 4: Optimise Syn term
                self.optim_VAE.zero_grad()
                syn_loss.backward()
                self.optim_VAE.step()  # Does the update in VAE network parameters

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

                # Saving traverse
                if not step % self.args.save_interval:
                    filename = 'alpha_' + str(self.alpha) + '_traversal_' + str(step) + '.png'
                    filepath = os.path.join(self.args.output_dir, filename)
                    traverse(self.net_mode, self.VAE, self.test_imgs, filepath)

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
        iters = torch.tensor(data['iter'])
        plot_metrics = {'recon': 'Reconstruction loss', 'kl': 'KL loss', 'syn': 'Synergy loss'}
        metrics_plot_dict = {k: torch.tensor(data[k]) for k in plot_metrics.keys()}
        latents_plot_dict = {f'count{i}': torch.tensor(data[f'l{i}']) for i in range(10)}

        # Metrics
        for plot, label in plot_metrics.items():
            self.viz.line(X=iters,
                          Y=metrics_plot_dict[plot],
                          win=self.win_id[plot],
                          update='append',
                          opts=dict(
                              xlabel='iteration',
                              ylabel=label))

        # Latents
        for i in range(10):
            self.viz.line(X=iters,
                          Y=latents_plot_dict[f'count{i}'],
                          env='/latents',
                          win=self.win_id[f'l{i}'],
                          update='append',
                          opts=dict(
                              xlabel='iteration',
                              ylabel='% Counts',
                              title=f'Latent {i}'))

    def viz_init(self):
        zero_init = torch.zeros([1])
        plot_metrics = {'recon': 'Reconstruction loss', 'kl': 'KL loss', 'syn': 'Synergy loss'}

        # Metrics
        for plot, label in plot_metrics.items():
            self.viz.line(X=zero_init,
                          Y=zero_init,
                          win=self.win_id[plot],
                          opts=dict(
                              xlabel='iteration',
                              ylabel=label,
                              title=label))

        # Latents
        for i in range(10):
            self.viz.line(X=zero_init,
                          Y=zero_init,
                          env='/latents',
                          win=self.win_id[f'l{i}'],
                          opts=dict(
                              xlabel='iteration',
                              ylabel='% Counts',
                              title=f'Latent {i}'))

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()