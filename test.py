import numpy as np
import torch
import random
from random import choice

from ops import kl_div_syn, kl_div, kl_div_uni_dim, kl_div_mean

# Lets going to try two different experiments: one per batch, one per sample

def I_max_ind(index, mu, logvar):

    mu_syn = mu[:, index]
    print(mu_syn)
    logvar_syn = logvar[:, index]
    print(logvar_syn)
    I_max = kl_div_syn(mu_syn, logvar_syn)
    print("normal KL {}".format(kl_div_syn(mu, logvar)))
    print("syn KL {}".format(I_max))

    return I_max


def I_max_batch(index, mu, logvar):

    mu_syn = mu[:, index]
    logvar_syn = logvar[:, index]

    print("this is the size {}".format(len(mu_syn.size())))

    if len(mu_syn.size()) == 1:
        I_max = kl_div_uni_dim(mu_syn, logvar_syn).mean()
        print("here")
    else:
        I_max = kl_div(mu_syn, logvar_syn)

    return I_max

def I_max_batch_mean(index, mu, logvar):

    mu_syn = mu[:, index]
    logvar_syn = logvar[:, index]

    print("this is the size {}".format(len(mu_syn.size())))

    if len(mu_syn.size()) == 1:
        I_max = kl_div_uni_dim(mu_syn, logvar_syn).mean()
        print("here")
    else:
        I_max = kl_div_mean(mu_syn, logvar_syn)

    return I_max


def generate_candidate(z_dim, best_c):

    if len(best_c) == 0:
        index = [i for i in range(0,10)]
    else:
        index = [i for i in range(0,10) if i not in best_c]

    return index

# KL of subsets ( same as paper)
def greedy_policy_Smax(z_dim, mu, logvar):

    best_c = []
    best_index = []
    Imax_best = 0

    for i in range(z_dim):
        print("z dim {}".format(i))
        index = generate_candidate(z_dim, best_c)
        print("this is index {}".format(index))

        for id in index:
            print("id: {}".format(id))
            c = best_index + [id]
            print("best index {}".format(best_index))
            print("c: {}".format(c))

            Imax_new = I_max_batch(c, mu, logvar)
            print("Imax_new {}".format(Imax_new))
            print("Imax_old {}".format(Imax_best))

            if Imax_new > Imax_best:
                print("update")
                best_c = c
                Imax_best = Imax_new

        best_index = best_c
        print(best_index)


    return best_index

# the same as with S_max but it uses a discount
def greedy_policy_Smax_discount(z_dim, mu, logvar, alpha):

    best_c = []
    best_index = []
    Imax_best = 0

    for i in range(z_dim):
        print("z dim {}".format(i))
        index = generate_candidate(z_dim, best_c)
        print("this is index {}".format(index))

        for id in index:
            print("id: {}".format(id))
            c = best_index + [id]
            print("best index {}".format(best_index))
            print("c: {}".format(c))

            Imax_new = I_max_batch(c, mu, logvar)
            print("Imax_new {}".format(Imax_new))
            print("Imax_old {}".format(Imax_best))
            print()

            if len(best_index) < 1:

                if Imax_new > Imax_best:
                    print("Update one dim, best_c {}, c{}, I_max_new {}, Imax_best {}".format(best_c,c,Imax_new,Imax_best))

                    best_c = c
                    Imax_best = Imax_new
            else:
                print("Imax_new = {}".format(Imax_new))
                print("Imax_new disc = {}".format(Imax_new * alpha))
                print("Imax_best = {}".format(Imax_best))
                if Imax_new * alpha > Imax_best:
                    print("Update more than one dim")
                    print("Update one dim, best_c {},c{}, I_max_new {}, Imax_best {}".format(best_c, c, Imax_new, Imax_best))

                    best_c = c
                    Imax_best = Imax_new

        best_index = best_c
        print(best_index)


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

            Imax_new = I_max_batch_mean(c, mu, logvar)
            print("Imax_new mean {}".format(Imax_new))
            print("Imax_new normal {}".format(I_max_batch(c,mu,logvar)))
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
        print("z dim {}".format(i))
        c = [i]
        print("c: {}".format(c))

        Imax_new = I_max_batch(c, mu, logvar)
        print("Imax_new {}".format(Imax_new))
        print("Imax_old {}".format(Imax_best))

        if Imax_new > Imax_best:
            print("update")
            best_c = c
            Imax_best = Imax_new
            print(best_c)

    print("final best_c {}".format(best_c))

    return best_c


mu = torch.Tensor([[0.4768, 0.8280, 0.1217, 0.7228, 0.7938, 0.5829, 0.0304, 0.7079, 0.2864,
         0.5885],
        [0.1685, 0.1480, 0.4624, 0.2667, 0.6145, 0.8437, 0.8690, 0.2979, 0.3222,
         0.7699],
        [0.6967, 0.9163, 0.2911, 0.7577, 0.3946, 0.8804, 0.0328, 0.2035, 0.5104,
         0.0499],
        [0.9070, 0.4434, 0.2247, 0.0418, 0.2147, 0.0525, 0.9763, 0.9890, 0.6643,
         0.8210],
        [0.9219, 0.5904, 0.3650, 0.9615, 0.5499, 0.9133, 0.4837, 0.7001, 0.3042,
         0.2848]])

logvar = torch.Tensor([[0.7089, 0.1473, 0.6645, 0.4639, 0.0684, 0.5317, 0.5295, 0.9676, 0.5997,
         0.6643],
        [0.0135, 0.6008, 0.0563, 0.6035, 0.7501, 0.3910, 0.0292, 0.6239, 0.8499,
         0.9242],
        [0.9635, 0.4796, 0.9727, 0.1286, 0.9708, 0.7271, 0.9971, 0.3047, 0.2433,
         0.4305],
        [0.2029, 0.0993, 0.1897, 0.4419, 0.4408, 0.9215, 0.4777, 0.5471, 0.5200,
         0.9320],
        [0.0848, 0.7966, 0.9400, 0.6471, 0.2608, 0.4046, 0.7163, 0.2900, 0.1172,
         0.8166]])



greedy_policy_Smax_discount(10, mu, logvar, 0.8)
a = kl_div(mu, logvar)
print(a)

