import numpy as np
import torch
import random
from random import choice

from utils.ops import kl_div_syn, kl_div, kl_div_uni_dim, kl_div_mean

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

    #print("this is the size {}".format(len(mu_syn.size())))

    if len(mu_syn.size()) == 1:
        I_max = kl_div_uni_dim(mu_syn, logvar_syn).mean()
        #print("here")
    else:
        #print(kl_div_uni_dim(mu_syn, logvar_syn))
        #print("MEAN")
        #print(kl_div_uni_dim(mu_syn, logvar_syn).mean())
        I_max = kl_div(mu_syn, logvar_syn)
        #print("IMAX NORMAL {}".format(I_max))

    return I_max

def I_max_batch_mean(index, mu, logvar):

    mu_syn = mu[:, index]
    logvar_syn = logvar[:, index]

    #print("this is the size {}".format(len(mu_syn.size())))

    if len(mu_syn.size()) == 1:
        I_max = kl_div_uni_dim(mu_syn, logvar_syn).mean()
        #print("here")
    else:
        I_max = kl_div_mean(mu_syn, logvar_syn)

    return I_max


def generate_candidate(z_dim, best_c):

    if len(best_c) == 0:
        index = [i for i in range(0,z_dim)]
    else:
        index = [i for i in range(0,z_dim) if i not in best_c]

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
        #print("z dim {}".format(i))
        index = generate_candidate(z_dim, best_c)
        #print("this is index {}".format(index))

        for id in index:
            #print("id: {}".format(id))
            c = best_index + [id]
            #print("best index {}".format(best_index))
            #print("c: {}".format(c))

            Imax_new = I_max_batch(c, mu, logvar)
            #print("Imax_new {}".format(Imax_new))
            #print("Imax_old {}".format(Imax_best))
            #print()

            if len(best_index) < 1:

                if Imax_new > Imax_best:
                    #print("Update one dim, best_c {}, c{}, I_max_new {}, Imax_best {}".format(best_c,c,Imax_new,Imax_best))

                    best_c = c
                    Imax_best = Imax_new
            else:
                #print("Imax_new = {}".format(Imax_new))
                #print("Imax_new disc = {}".format(Imax_new * alpha))
                #print("Imax_best = {}".format(Imax_best))
                if Imax_new * alpha > Imax_best:
                    #print("Update more than one dim")
                    #print("Update one dim, best_c {},c{}, I_max_new {}, Imax_best {}".format(best_c, c, Imax_new, Imax_best))

                    best_c = c
                    Imax_best = Imax_new

        best_index = best_c
        #print(best_index)


    return best_index


# the same as with S_max but it uses a discount
def greedy_policy_Smax_discount_worst(z_dim, mu, logvar, alpha):

    best_c = []
    best_index = []
    Imax_best = 0
    worst_index = []

    for i in range(z_dim):
        #print("z dim {}".format(i))
        index = generate_candidate(z_dim, best_c)
        #print("this is index {}".format(index))

        for id in index:
            #print("id: {}".format(id))
            c = best_index + [id]
            #print("best index {}".format(best_index))
            #print("c: {}".format(c))

            Imax_new = I_max_batch(c, mu, logvar)
            #print("Imax_new {}".format(Imax_new))
            #print("Imax_old {}".format(Imax_best))
            #print()

            if len(best_index) < 1:

                if Imax_new > Imax_best:
                    #print("Update one dim, best_c {}, c{}, I_max_new {}, Imax_best {}".format(best_c,c,Imax_new,Imax_best))

                    best_c = c
                    Imax_best = Imax_new
            else:
                #print("Imax_new = {}".format(Imax_new))
                #print("Imax_new disc = {}".format(Imax_new * alpha))
                #print("Imax_best = {}".format(Imax_best))
                if Imax_new * alpha > Imax_best:
                    #print("Update more than one dim")
                    #print("Update one dim, best_c {},c{}, I_max_new {}, Imax_best {}".format(best_c, c, Imax_new, Imax_best))

                    best_c = c
                    Imax_best = Imax_new

        best_index = best_c
        #print(best_index)
    worst_index = [i for i in range(0,z_dim) if i not in best_index]


    return best_index, worst_index



def e_greedy_policy_Smax_discount(z_dim, mu, logvar, alpha, epsilon):

    best_c = []
    best_index = []
    Imax_best = 0

    for i in range(z_dim):
        #print("z dim {}".format(i))
        index = generate_candidate(z_dim, best_c)
        #print("this is index {}".format(index))

        p = np.random.uniform(0,1)
        #print("value of p {}, dim {}".format(p, i))

        if p <= epsilon:
            #print("e-greedy, dim {}".format(i))
            indices = list(set(list(range(0,z_dim))) - set(best_c))
            best_c = [random.choice(indices)] + best_c
            Imax_best = I_max_batch(best_c, mu, logvar)
            #print("c, egreedy {}".format(best_c))
            #print("Imax_best, egreedy {}".format(Imax_best))
            #print()
            best_index = best_c

        else:

            for id in index:
                #print("id: {}".format(id))
                c = best_index + [id]
                #print("best index {}".format(best_index))
                #print("c: {}".format(c))

                Imax_new = I_max_batch(c, mu, logvar)
                #print("Imax_new {}".format(Imax_new))
                #print("Imax_old (best) {}".format(Imax_best))
                #print()


                if len(best_index) < 1:

                    if Imax_new > Imax_best:
                        #print("Update one dim, best_c {}, c{}, I_max_new {}, Imax_best {}".format(best_c,c,Imax_new,Imax_best))

                        best_c = c
                        Imax_best = Imax_new
                else:
                    #print("Imax_new = {}".format(Imax_new))
                    #print("Imax_new disc = {}".format(Imax_new * alpha))
                    #print("Imax_best = {}".format(Imax_best))
                    if Imax_new * alpha > Imax_best:
                        #print("Update more than one dim")
                        #print("Update one dim, best_c {},c{}, I_max_new {}, Imax_best {}".format(best_c, c, Imax_new, Imax_best))

                        best_c = c
                        Imax_best = Imax_new

            best_index = best_c

        #print(best_index)

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
        #print("z dim {}".format(i))
        c = [i]
        #print("c: {}".format(c))

        Imax_new = I_max_batch(c, mu, logvar)
        #print("Imax_new {}".format(Imax_new))
        #print("Imax_old {}".format(Imax_best))

        if Imax_new > Imax_best:
            #print("update")
            best_c = c
            Imax_best = Imax_new
            #print(best_c)

    #print("final best_c {}".format(best_c))

    return best_c

# Compute KL element wise
def e_greedy_policy_one_dim(z_dim, mu, logvar, epsilon):
    best_c = []
    Imax_best = 0

    p = np.random.uniform(0, 1)
    #print("value of p {}".format(p))

    if p <= epsilon:
        #print("e-greedy")
        best_c = [random.choice(range(0,10))]

    else:
        for i in range(z_dim):
            #print("z dim {}".format(i))
            c = [i]
            #print("c: {}".format(c))

            Imax_new = I_max_batch(c, mu, logvar)
            #print("Imax_new {}".format(Imax_new))
            #print("Imax_old {}".format(Imax_best))

            if Imax_new > Imax_best:
                #print("update")
                best_c = c
                Imax_best = Imax_new
                #print(best_c)

        #print("final best_c {}".format(best_c))
    #print(best_c)
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


logvar1 = torch.tensor([[-0.8235, -4.0874, -1.6253, -1.8602, -1.5565, -3.2423, -2.1147, -3.0013,
         -3.5243, -2.0131],
        [-1.0157, -3.0389, -1.2551, -2.2479, -1.1635, -2.2660, -2.4355, -2.9129,
         -2.7966, -1.1832],
        [-1.1491, -3.7570, -1.4187, -2.2754, -1.3702, -2.8456, -2.6524, -3.4643,
         -3.6787, -1.6336],
        [-0.0065, -2.9606, -0.9316, -1.3927, -1.4184, -2.4480, -2.0936, -2.8926,
         -2.9241, -1.3678],
        [-0.3606, -3.3903, -1.0816, -1.5726, -1.4835, -2.8189, -2.3822, -3.4395,
         -3.4242, -1.4950],
        [-0.8249, -3.6379, -1.5062, -2.1452, -1.5382, -2.9740, -2.2288, -2.8659,
         -2.8932, -1.8953],
        [-0.8994, -3.3491, -1.3738, -1.1361, -1.1162, -2.4299, -2.2719, -2.8495,
         -4.0788, -1.5827],
        [-1.0358, -3.1172, -1.2872, -2.3263, -1.1946, -2.3060, -2.4784, -2.9490,
         -2.8359, -1.2427],
        [-0.8715, -3.5234, -1.3418, -2.2838, -1.3958, -2.7802, -2.4634, -3.1604,
         -3.0275, -1.5786],
        [-0.7882, -3.6559, -1.3956, -2.1666, -1.4652, -2.9239, -2.2825, -3.0045,
         -2.9884, -1.7599],
        [-1.2402, -3.3400, -1.4979, -2.4686, -1.2118, -2.3637, -2.4803, -2.9099,
         -3.1027, -1.4653],
        [-0.4291, -3.4946, -1.5213, -1.2622, -1.6036, -2.3445, -1.8954, -2.8933,
         -4.4206, -1.9528],
        [-1.3247, -3.0542, -1.2198, -1.4573, -0.8737, -2.2551, -2.3154, -2.6392,
         -3.4788, -1.1332],
        [-0.3269, -3.0445, -1.0852, -0.9776, -1.3123, -2.3363, -2.0638, -2.9830,
         -3.8055, -1.5150],
        [-0.3147, -3.0532, -1.0567, -1.0319, -1.3163, -2.4364, -2.1390, -3.1179,
         -3.6820, -1.4594],
        [-0.2800, -3.0917, -1.0211, -1.7809, -1.4273, -2.5926, -2.2409, -3.0453,
         -2.7985, -1.3640],
        [-0.8251, -3.1889, -1.2296, -2.1583, -1.2844, -2.4860, -2.3151, -2.9002,
         -2.6549, -1.3791],
        [ 0.0015, -2.9130, -1.0757, -0.8840, -1.3844, -2.1206, -1.7033, -2.6028,
         -3.8719, -1.6946],
        [-0.0869, -2.8696, -0.9166, -1.4821, -1.3774, -2.3823, -2.0804, -2.7785,
         -2.6404, -1.2863],
        [-1.4072, -3.5845, -1.6935, -2.5197, -1.2752, -2.4267, -2.4777, -2.9680,
         -3.4621, -1.6307],
        [-0.6450, -2.6288, -0.9356, -0.9049, -0.9028, -1.9813, -2.0954, -2.4063,
         -3.3188, -1.1102],
        [-0.7575, -4.0705, -1.5960, -1.8271, -1.5989, -3.2360, -2.2088, -3.1852,
         -3.6954, -1.9828],
        [-1.5633, -3.5447, -1.6057, -2.4183, -1.1128, -2.3153, -2.4599, -3.0168,
         -3.6450, -1.5922],
        [-0.1627, -3.3636, -1.1642, -1.3334, -1.6278, -2.6452, -1.9504, -3.0322,
         -3.8339, -1.6900],
        [-1.3977, -3.7693, -1.6029, -1.6215, -1.1080, -2.7225, -2.4181, -3.0053,
         -4.2154, -1.6382],
        [ 0.0241, -2.9586, -1.0040, -1.0587, -1.4402, -2.3226, -1.8135, -2.6681,
         -3.3854, -1.5188],
        [-0.0980, -3.1803, -1.1530, -1.0872, -1.5686, -2.3733, -1.7477, -2.8013,
         -4.0078, -1.7188],
        [-0.9509, -4.2039, -1.7193, -1.9116, -1.5577, -3.3282, -2.0867, -2.9102,
         -3.5836, -2.1483],
        [-0.3985, -3.1443, -1.2503, -1.0097, -1.3661, -2.2197, -1.9592, -2.7930,
         -4.1267, -1.6568],
        [-0.4026, -3.5907, -1.2556, -1.4893, -1.6327, -2.8739, -2.2129, -3.4023,
         -3.9235, -1.7022],
        [-0.0552, -2.9946, -1.0373, -0.9898, -1.4809, -2.2738, -1.7859, -2.7982,
         -3.8206, -1.6071],
        [-1.0748, -3.5893, -1.5480, -1.2359, -1.1532, -2.5524, -2.3399, -2.9285,
         -4.2764, -1.7034],
        [-0.3925, -3.1412, -1.0097, -1.7624, -1.3637, -2.6077, -2.3568, -3.1293,
         -2.8779, -1.3179],
        [-1.2996, -3.0325, -1.2553, -1.3480, -0.8404, -2.1960, -2.2829, -2.5361,
         -3.5265, -1.2029],
        [-0.2080, -3.4197, -1.2176, -1.6015, -1.5531, -2.7925, -1.9164, -2.6963,
         -3.2358, -1.7785],
        [-1.4084, -3.3719, -1.5002, -1.3241, -0.9211, -2.3732, -2.2694, -2.5585,
         -3.9452, -1.4353],
        [-1.2893, -2.9396, -1.2240, -1.3693, -0.8190, -2.1203, -2.2726, -2.4626,
         -3.4696, -1.1129],
        [-1.3076, -4.1995, -1.7501, -2.4743, -1.5010, -3.0506, -2.5896, -3.3329,
         -3.9642, -1.9831],
        [-1.3599, -3.1328, -1.3419, -2.0505, -1.0189, -2.2501, -2.4253, -2.8619,
         -3.2496, -1.1426],
        [-1.5025, -3.5638, -1.4670, -1.8104, -1.0232, -2.5540, -2.4198, -2.8923,
         -3.8709, -1.4252],
        [-0.4499, -3.7169, -1.3654, -1.5227, -1.6799, -2.9449, -2.1616, -3.3916,
         -4.0772, -1.8246],
        [-0.9064, -3.5998, -1.4555, -1.3386, -1.2752, -2.6803, -2.3846, -3.1918,
         -4.2375, -1.6817],
        [-1.6536, -4.0355, -1.7832, -1.8280, -1.1256, -2.8007, -2.4152, -2.9450,
         -4.3405, -1.7569],
        [-0.6797, -3.4888, -1.3592, -1.2761, -1.3712, -2.6023, -2.2908, -3.2063,
         -4.1864, -1.6766],
        [-0.3930, -3.1960, -1.2858, -1.0513, -1.4165, -2.2538, -1.9410, -2.8217,
         -4.1660, -1.6948],
        [-0.1858, -3.2169, -1.1582, -1.1171, -1.5779, -2.4339, -1.8491, -2.9368,
         -3.9556, -1.6706],
        [-0.9284, -3.5969, -1.3828, -2.3498, -1.4185, -2.8145, -2.5119, -3.1891,
         -3.1179, -1.6163],
        [-0.4546, -3.1343, -1.1476, -1.0375, -1.2867, -2.3563, -2.1271, -2.9864,
         -3.9285, -1.5473],
        [-0.2973, -3.3062, -1.2704, -1.1530, -1.5691, -2.3938, -1.8633, -2.8957,
         -4.1436, -1.7499],
        [-1.3312, -3.4000, -1.4766, -2.3668, -1.1690, -2.4225, -2.4676, -3.0419,
         -3.3123, -1.4988],
        [-1.1621, -4.0026, -1.6461, -2.4884, -1.4648, -2.9864, -2.4966, -3.1327,
         -3.4716, -1.9002],
        [-0.5456, -3.4685, -1.4505, -1.2420, -1.5244, -2.3704, -2.0084, -2.9295,
         -4.4159, -1.8123],
        [-0.6734, -3.9414, -1.5608, -1.7790, -1.5877, -3.2090, -2.0182, -2.8880,
         -3.4366, -2.0421],
        [-1.1886, -4.1946, -1.6731, -2.4546, -1.5134, -3.1765, -2.5799, -3.3144,
         -3.7749, -1.9585],
        [-1.6956, -4.0062, -1.8306, -1.7258, -1.0851, -2.7470, -2.3625, -2.8178,
         -4.3295, -1.7534],
        [-0.7224, -3.8885, -1.5545, -1.8798, -1.5718, -3.1700, -2.0360, -2.8318,
         -3.2676, -2.0045],
        [-0.7089, -3.9548, -1.5723, -1.7937, -1.5932, -3.1980, -1.9915, -2.8428,
         -3.4437, -2.0609],
        [-0.2425, -3.4689, -1.2156, -1.3790, -1.6474, -2.7444, -2.0096, -3.1467,
         -3.9360, -1.7257],
        [-0.8919, -3.5351, -1.4839, -1.1999, -1.2318, -2.5208, -2.3021, -2.9820,
         -4.3133, -1.7050],
        [-1.1781, -4.2627, -1.7222, -2.2429, -1.5188, -3.2660, -2.3155, -3.0633,
         -3.5453, -2.0440],
        [-0.1979, -3.4379, -1.2252, -1.3616, -1.6558, -2.6878, -1.9055, -3.0010,
         -3.8852, -1.7544],
        [-0.6559, -3.9470, -1.5578, -1.7296, -1.5769, -3.1631, -2.0417, -2.9660,
         -3.5295, -1.9708],
        [-1.0001, -2.7438, -1.1158, -1.7505, -1.0215, -2.1162, -2.3602, -2.7162,
         -2.8141, -0.8934],
        [-0.7473, -3.8047, -1.4505, -2.0267, -1.5239, -3.0817, -2.2098, -2.9667,
         -3.0928, -1.8466]], requires_grad = True)

mu1 = torch.tensor([[-0.3549,  4.0147, -0.8859, -0.5778, -0.3731, -1.2710, -3.2846, -0.7766,2.8586, -1.7056],
                    [ 0.2670,  1.8532,  0.7425, -0.7004, -0.1404,  3.6998, -2.7002, -3.2500,2.6169,  0.1248],
                    [ 0.0729,  2.7594,  0.0004, -0.5892, -0.7642,  2.5332, -3.8715, -3.1429,
                      2.1170,  0.8286],
                    [ 0.7190,  1.2263,  0.8272, -2.2644,  0.0435, -1.4281, -3.0755, -3.4950,
                      1.8984, -1.0314],
                    [ 0.6625,  1.7881,  0.5445, -1.9686, -0.4799, -0.4396, -3.9314, -3.9556,
                      1.5234,  0.2101],
                    [ 0.0615,  3.4025,  0.1148, -1.3968,  0.0440,  1.0622, -2.4180, -1.1153,
                      3.2707, -2.3945],
                    [-0.2733,  1.1587, -1.0703, -0.8434, -2.2562, -1.0257, -5.1604, -1.5689,
                      1.4675,  2.6928],
                    [ 0.2281,  2.0711,  0.7261, -0.6295, -0.1053,  3.7734, -2.6033, -3.1735,
                      2.7648,  0.0078],
                    [ 0.2764,  2.9184,  0.5487, -1.2421, -0.1561,  2.3348, -2.8521, -2.7349,
                      2.5511, -0.8090],
                    [ 0.0945,  3.4269,  0.2079, -1.2188, -0.0784,  1.1105, -2.6176, -1.8762,
                      2.8206, -1.6902],
                    [-0.1711,  2.9616,  0.2011,  0.3507, -0.3038,  3.5196, -2.3585, -2.5647,
                      3.1059,  0.2516],
                    [-0.6947,  2.4084, -1.3805,  1.1855, -1.7042, -4.6484, -3.0456, -1.9480,
                      2.6342,  2.0750],
                    [-0.2262,  0.4267, -0.6300, -0.7573, -1.7586,  2.5181, -5.1080, -1.6088,
                      1.5320,  2.4097],
                    [ 0.1735,  1.1332, -0.3136, -1.0929, -1.5470, -2.5710, -4.2087, -3.0050,
                      1.3283,  1.9513],
                    [ 0.3863,  1.0785, -0.0575, -1.4776, -1.3467, -2.0160, -4.3578, -3.3314,
                      1.1787,  1.6999],
                    [ 0.7349,  1.8748,  0.9471, -2.1140,  0.1360,  0.2691, -2.9116, -3.6293,
                      1.9959, -1.1110],
                    [ 0.3584,  2.1995,  0.7576, -1.4727,  0.0962,  2.7601, -2.6248, -2.7414,
                      2.6498, -1.1711],
                    [-0.0611,  0.9485, -0.2905, -0.1104, -1.1396, -4.9575, -2.8160, -2.8003,
                      2.2030,  1.4936],
                    [ 0.7840,  1.0440,  1.0344, -2.5706,  0.2356, -0.4909, -2.8799, -3.3185,
                      2.0686, -1.5116],
                    [-0.5216,  3.5658, -0.3226,  1.2770, -0.5909,  2.8647, -2.3424, -2.1796,
                      3.3737,  0.7106],
                    [ 0.2424, -0.6383,  0.1153, -1.6937, -1.6338, -0.3816, -4.5802, -2.3405,
                      1.3991,  2.2617],
                    [-0.2336,  3.9141, -0.8225, -0.5487, -0.4954, -1.5337, -3.4903, -1.4118,
                      2.5999, -1.1225],
                    [-0.7253,  3.0155, -0.7138,  1.5394, -1.0188,  3.1397, -3.0802, -2.2076,
                      2.9616,  1.5934],
                    [ 0.1650,  2.3871, -0.3779, -0.4267, -0.5801, -3.6610, -3.1431, -2.9261,
                      2.1482,  0.0514],
                    [-0.5774,  2.0898, -1.5163, -0.0952, -2.1119,  0.8850, -5.3605, -0.9634,
                      1.7839,  2.3908],
                    [ 0.3131,  1.3925,  0.0205, -1.1485, -0.4387, -3.6248, -3.0158, -2.7597,
                      2.0677, -0.2653],
                    [-0.1261,  2.0141, -0.6144,  0.2744, -1.0067, -4.8123, -2.7886, -2.7640,
                      2.2731,  1.0079],
                    [-0.5005,  4.1292, -1.1236, -0.4469, -0.4062, -1.1922, -3.2562, -0.1230,
                      3.1882, -2.0039],
                    [-0.2991,  1.5175, -0.9263, -0.0328, -1.8406, -3.6446, -3.7738, -2.2919,
                      1.8199,  2.3570],
                    [ 0.2539,  2.6105, -0.3154, -0.7247, -0.8286, -2.2963, -3.8269, -3.3032,
                      1.8087,  0.6399],
                    [ 0.0880,  1.5192, -0.3140, -0.2722, -1.0178, -4.2895, -3.0134, -3.0937,
                      1.9280,  1.1948],
                    [-0.4772,  1.6267, -1.4482, -0.4976, -2.4109, -0.8547, -5.3151, -1.0866,
                      1.6146,  2.7712],
                    [ 0.7747,  1.5137,  1.0065, -2.3837, -0.0610,  1.0107, -3.3734, -3.7423,
                      1.8601, -0.6852],
                    [-0.3362,  0.3395, -0.8089, -0.7321, -1.9711,  1.9905, -5.1523, -1.2201,
                      1.5514,  2.6090],
                    [ 0.2362,  2.4359, -0.0563, -1.2548,  0.1049, -2.5290, -2.7549, -1.6979,
                      2.9562, -2.1537],
                    [-0.6447,  1.0286, -1.5473, -0.2528, -2.3750,  0.9730, -5.4161, -0.1453,
                      1.7354,  2.6689],
                    [-0.3190,  0.0523, -0.7028, -0.6846, -1.9260,  2.1925, -5.0668, -1.3293,
                      1.6307,  2.6145],
                    [-0.5283,  4.1872, -0.7472,  0.4543, -1.0050,  1.4138, -3.3310, -1.7268,
                      2.9471,  0.3465],
                    [-0.1316,  1.4016, -0.1170, -0.0415, -0.9378,  3.7967, -3.8819, -2.6431,
                      2.2164,  1.5550],
                    [-0.5206,  1.7257, -1.1592, -0.0003, -1.7579,  2.4763, -5.0799, -1.3693,
                      1.8519,  2.1955],
                    [ 0.0715,  3.0691, -0.6529, -0.2215, -0.8673, -2.8276, -3.6796, -2.9373,
                      2.0155,  0.5534],
                    [-0.2481,  1.8794, -1.0848, -0.6882, -2.0610, -0.9945, -5.1025, -1.9658,
                      1.5140,  2.4182],
                    [-0.9209,  2.6563, -1.9543,  0.7078, -2.1395,  1.3259, -5.2258, -0.2722,
                      2.2270,  2.2165],
                    [-0.1599,  1.8850, -0.8800, -0.6257, -1.8722, -1.7999, -4.6629, -2.4389,
                      1.5559,  2.2355],
                    [-0.3492,  1.7293, -1.0061,  0.1672, -1.7996, -3.8194, -3.6374, -2.2612,
                      1.9308,  2.2626],
                    [-0.0515,  2.1663, -0.6112,  0.0274, -1.1308, -4.1227, -3.1613, -2.9247,
                      1.9600,  1.1862],
                    [ 0.2160,  3.0654,  0.4864, -1.0910, -0.2451,  2.4696, -2.8497, -2.6530,
                      2.6483, -0.7044],
                    [ 0.0355,  1.1864, -0.5185, -0.9681, -1.7567, -2.3012, -4.4194, -2.7500,
                      1.3762,  2.2175],
                    [-0.3183,  2.3104, -0.9775,  0.4984, -1.4057, -4.3139, -3.1699, -2.4919,
                      2.1761,  1.5782],
                    [-0.2367,  2.7566, -0.0693,  0.4031, -0.6410,  3.3941, -3.0050, -2.7280,
                      2.6439,  0.9020],
                    [-0.3122,  4.0926, -0.2546, -0.1129, -0.5726,  1.8269, -2.7618, -1.6051,
                      3.1702, -0.6483],
                    [-0.5838,  2.3807, -1.4056,  0.7557, -1.8805, -3.7977, -3.5907, -1.9498,
                      2.2612,  2.2410],
                    [-0.2010,  3.6058, -0.7512, -0.8477, -0.1929, -1.7133, -3.1349, -0.7118,
                      3.0308, -2.1359],
                    [-0.3292,  4.1878, -0.4709, -0.2074, -0.7923,  1.4293, -3.2664, -1.6928,
                      2.8992, -0.3341],
                    [-1.0082,  2.5731, -2.1171,  0.8110, -2.2923,  1.1733, -5.2893,  0.1956,
                      2.2485,  2.2862],
                    [-0.1738,  3.6433, -0.5697, -0.9549, -0.1229, -1.0158, -2.8568, -0.6835,
                      3.1270, -2.3244],
                    [-0.2715,  3.7377, -0.8395, -0.6573, -0.2061, -1.8029, -3.0174, -0.5481,
                      3.1237, -2.1695],
                    [ 0.1411,  2.5966, -0.4855, -0.3279, -0.6726, -3.5016, -3.3160, -2.9261,
                      2.0854,  0.2364],
                    [-0.4092,  1.7023, -1.2925, -0.4559, -2.3055, -1.5817, -5.0161, -1.5451,
                      1.6419,  2.7065],
                    [-0.4920,  4.4346, -0.7406, -0.4226, -0.5773,  0.5874, -3.0608, -0.6725,
                      3.1466, -1.4771],
                    [ 0.0547,  2.6453, -0.5636, -0.1658, -0.5690, -3.9038, -3.0293, -2.6093,
                      2.3259, -0.1051],
                    [-0.2466,  3.7979, -0.8642, -0.5702, -0.3069, -2.0702, -3.2485, -1.0027,
                      2.7889, -1.6661],
                    [ 0.3559,  0.2904,  0.6484, -1.2969, -0.5849,  3.4464, -3.7077, -3.1249,
                      2.0400,  0.9923],
                    [ 0.0006,  3.5547, -0.1056, -1.3425, -0.0948,  0.2453, -2.8294, -1.3805,
                      2.8315, -2.0763]], requires_grad = True)


#greedy_policy_Smax_discount(10, mu, logvar, 0.8)
#a = kl_div(mu, logvar)
#print(a)

#greedy_policy_Smax(10, mu1, logvar1)


#for i in np.arange(0.6,1,0.01):

#    a = greedy_policy_Smax_discount(10, mu1, logvar1, i)
#    print(a, i)

#metric_1A(mu1, logvar1, 10, 64)

#e_greedy_policy_one_dim(10, mu1, logvar1, 0.5)



