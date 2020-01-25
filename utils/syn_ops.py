import numpy as np
import torch
import random

from utils.ops import kl_div_syn, kl_div, kl_div_uni_dim, kl_div_mean
# Lets going to try two different experiments: one per batch, one per sample

def i_max_ind(index, mu, log_var):

    mu_syn = mu[:, index]
    log_var_syn = log_var[:, index]
    i_max = kl_div_syn(mu_syn, log_var_syn)

    return i_max


def i_max_batch(index, mu, log_var):

    mu_syn = mu[:, index]
    log_var_syn = log_var[:, index]

    if len(mu_syn.size()) == 1:
        i_max = kl_div_uni_dim(mu_syn, log_var_syn).mean()
    else:
        i_max = kl_div(mu_syn, log_var_syn)

    return i_max


def i_max_batch_mean(index, mu, log_var):

    mu_syn = mu[:, index]
    log_var_syn = log_var[:, index]

    if len(mu_syn.size()) == 1:
        i_max = kl_div_uni_dim(mu_syn, log_var_syn).mean()
    else:
        i_max = kl_div_mean(mu_syn, log_var_syn)
    return i_max


def generate_candidate(z_dim, best_c):

    if len(best_c) == 0:
        index = [i for i in range(0, z_dim)]
    else:
        index = [i for i in range(0, z_dim) if i not in best_c]
    return index


# KL of subsets ( same as paper)
def greedy_policy_s_max(z_dim, mu, log_var):

    best_c = []
    best_index = []
    Imax_best = 0

    for i in range(z_dim):
        print("z dim {}".format(i))
        index = generate_candidate(z_dim, best_c)
        print("this is index {}".format(index))

        for id in index:
            c = best_index + [id]
            print("best index {}".format(best_index))
            print("c: {}".format(c))

            Imax_new = i_max_batch(c, mu, log_var)
            print("Imax_new {}".format(Imax_new))
            print("Imax_old {}".format(Imax_best))

            if Imax_new > Imax_best:
                best_c = c
                Imax_best = Imax_new

        best_index = best_c
        print(best_index)

    return best_index


# the same as with S_max but it uses a discount
def greedy_policy_Smax_discount(z_dim, mu, log_var, alpha):

    best_c = []
    best_index = []
    i_max_best = 0

    for i in range(z_dim):

        index = generate_candidate(z_dim, best_c)
        for id in index:

            c = best_index + [id]
            i_max_new = i_max_batch(c, mu, log_var)

            if len(best_index) < 1:

                if i_max_new > i_max_best:
                    best_c = c
                    i_max_best = i_max_new
            else:
                if i_max_new * alpha > i_max_best:
                    best_c = c
                    i_max_best = i_max_new
        best_index = best_c

    return best_index


# the same as with S_max but it uses a discount
def greedy_policy_s_max_discount_worst(z_dim, mu, log_var, alpha):

    best_c = []
    best_index = []
    i_max_best = 0

    for i in range(z_dim):
        index = generate_candidate(z_dim, best_c)

        for id in index:
            c = best_index + [id]
            i_max_new = i_max_batch(c, mu, log_var)

            if len(best_index) < 1:

                if i_max_new > i_max_best:
                    best_c = c
                    i_max_best = i_max_new
            else:
                if i_max_new * alpha > i_max_best:
                    best_c = c
                    i_max_best = i_max_new

        best_index = best_c

    worst_index = [i for i in range(0, z_dim) if i not in best_index]
    return best_index, worst_index


def e_greedy_policy_Smax_discount(z_dim, mu, logvar, alpha, epsilon):

    best_c = []
    best_index = []
    Imax_best = 0

    for i in range(z_dim):
        index = generate_candidate(z_dim, best_c)
        p = np.random.uniform(0,1)

        if p <= epsilon:
            indices = list(set(list(range(0,z_dim))) - set(best_c))
            best_c = [random.choice(indices)] + best_c
            Imax_best = i_max_batch(best_c, mu, logvar)
            best_index = best_c

        else:
            for id in index:

                c = best_index + [id]
                Imax_new = i_max_batch(c, mu, logvar)
                if len(best_index) < 1:

                    if Imax_new > Imax_best:
                        best_c = c
                        Imax_best = Imax_new
                else:
                    if Imax_new * alpha > Imax_best:
                        best_c = c
                        Imax_best = Imax_new

            best_index = best_c

    return best_index


# Instead of summing compute the mean of the KL along the dim
# I want to use an RL approach using a simple multi bandit problem.
def greedy_policy_Igs(z_dim, mu, logvar):

    best_c = []
    best_index = []
    Imax_best = 0
    count_dim = 0

    for i in range(z_dim):
        print("z dim {}".format(i))
        index = generate_candidate(z_dim, best_c)
        print("this is index {}".format(index))
        count_dim =+ 1

        for id in index:
            print("id: {}".format(id))
            c = best_index + [id]

            print("best index {}".format(best_index))
            print("c: {}".format(c))
            print("size of c: {}".format(len(c)))

            Imax_new = i_max_batch_mean(c, mu, logvar)
            print("Imax_new mean {}".format(Imax_new))
            print("Imax_new normal {}".format(i_max_batch(c,mu,logvar)))
            print("Imax_old {}".format(Imax_best))

            if Imax_new > Imax_best:
                print("update")
                best_c = c
                Imax_best = Imax_new

        best_index = best_c
        print("best_index {}, count_dim {}".format(best_index, count_dim))
        if not best_index == count_dim:
            break
        print(best_index)

    print("result {}".format(best_index))

    return best_index


# Compute KL element wise
def greedy_policy_one_dim(z_dim, mu, logvar):

    best_c = []
    Imax_best = 0

    for i in range(z_dim):
        c = [i]

        Imax_new = i_max_batch(c, mu, logvar)

        if Imax_new > Imax_best:
            best_c = c
            Imax_best = Imax_new

    return best_c


# Compute KL element wise
def e_greedy_policy_one_dim(z_dim, mu, logvar, epsilon):
    best_c = []
    Imax_best = 0

    p = np.random.uniform(0, 1)
    if p <= epsilon:
        best_c = [random.choice(range(0,10))]

    else:
        for i in range(z_dim):
            c = [i]
            Imax_new = i_max_batch(c, mu, logvar)

            if Imax_new > Imax_best:
                best_c = c
                Imax_best = Imax_new

    return best_c


def S_metric_1A(mu, logvar, z_dim, batch_size):

    alpha = 1.5
    Smax = torch.empty((1,batch_size))

    for s in range(batch_size):

        mu_s = mu[s,:].view(1,-1)
        logvar_s = logvar[s,:].view(1,-1)

        # get the argmax
        index = greedy_policy_Smax_discount(z_dim, mu_s,logvar_s,alpha=0.8)
        print("sample {}, index {}".format(s, index))

        # get the dims:
        mu_syn = mu_s[:, index]
        logvar_syn = logvar_s[:, index]

        if len(mu_syn.size()) == 1:
            I_m = kl_div_uni_dim(mu_syn, logvar_syn).mean()
                            # print("here")
        else:
            I_m = kl_div(mu_syn, logvar_syn)

        Smax[0,s] = I_m

    print("Smax {}".format(Smax))
    print("Smax size {}".format(Smax.size()))
    print("Smax requires grad {}".format(Smax.requires_grad))
    I_max= Smax.mean()
    print("I_max {}".format(I_max))
    print("I_max {}".format(I_max.requires_grad))

    syn_loss = alpha * I_max

    return syn_loss



