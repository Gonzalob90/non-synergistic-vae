
import torch
import numpy as np
import random
from collections import defaultdict, Counter


VAR_THRESHOLD = 1e-2

# Load dataset
dataset_zip = np.load('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='latin1')

imgs = dataset_zip['imgs']
latents_values = dataset_zip['latents_values']
latents_classes = dataset_zip['latents_classes']
metadata = dataset_zip['metadata'][()]

# Define number of values per latents and functions to convert to indices
latents_sizes = metadata['latents_sizes']
# array([ 1,  3,  6, 40, 32, 32])

latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))



def latent_to_index(latents):
    return np.dot(latents, latents_bases).astype(int)


def sample_latent(size=1):

    samples = np.zeros((size, latents_sizes.size))
    for lat_i, lat_size in enumerate(latents_sizes):
        samples[:, lat_i] = np.random.randint(lat_size, size=size)
    return samples


def gini_variance(qz_samples_norm, active_latents_index):

    L = 100
    gini = {a.item(): [] for a in active_latents_index}

    for dim in active_latents_index:

        d = 0

        #print(len(qz_samples_norm[dim.item()]))
        #print(qz_samples_norm[dim.item()])

        for i in qz_samples_norm[dim.item()]:
            for j in qz_samples_norm[dim.item()]:

                #print("i {}, j {}".format(i,j))

                d += (i - j).pow(2).item()
                #print(d)

        #print("dim {}, d {}".format(dim, d))

        var = d / (2 * L * (L-1))
        gini[dim.item()] = var

    return gini


def factor_vae_metric_shapes(train_model, model, votes = 800):

    train_model(train=False)

    L = 100
    M = 500
    D = model.z_dim
    list_votes = []

    for v in range(votes):

        if v % 20 == 0:
            print("vote {}".format(v))

        training_set = []

        for i in range(M) :

            #print("vote {}, i {}".format(v,i))

            # fix one factor
            #sample a dimension
            k_dim = random.sample(range(latents_sizes.shape[0]), 1)[0]
            # sample the factor k
            factor_k = random.sample(range(latents_sizes[k_dim]), 1)[0]

            #print("k_dim {}".format(k_dim))
            #print("factor_k {}".format(factor_k))

            ## Sample using the factor k
            latents_sampled = sample_latent(size=L)
            latents_sampled[:, k_dim] = factor_k
            indices_sampled = latent_to_index(latents_sampled)
            imgs_sampled = imgs[indices_sampled]
            # shape of imgs_sampled (L, 64, 64)

            xs = torch.from_numpy(imgs_sampled)
            #print("xs size {}".format(xs.size()))
            #print("xs type {}".format(xs.type()))

            xs = xs.type(torch.float)
            #print("xs type {}".format(xs.type()))

            xs_sampled = xs.unsqueeze(1).to("cuda")
            #xs_sampled = xs.unsqueeze(1).to("cpu")
            #print("xs sampled {}".format(xs_sampled.size()))

            # Sample Z
            qz_means = model(xs_sampled, decode=False)[1]
            qz_samples = model(xs_sampled, decode=False)[3]
            #print("qz_means size {}".format(qz_means.size()))

            qz_samples = qz_samples.cuda()
            #qz_samples = qz_samples.cpu()
            # qz_samples shape (L, D)

            # Compute the standard deviation
            s = torch.std(qz_samples, dim = 0)
            #print("s {}".format(s))
            #print("s size {} ".format(s.size()))

            # Normalise
            qz_samples_norm = qz_samples / s
            # print("qz samples norm size {}".format(qz_samples_norm.size()))
            # qz_samples_norm shape (100,10)

            # prune active units
            var = torch.std(qz_means.contiguous().view(L, D), dim=0).pow(2)  # pow is just **2, contiguous in memory
            active_units = torch.arange(0, D)[var > VAR_THRESHOLD].long()
            #print('Active units: ' + ','.join(map(str, active_units.tolist())))

            #if len(active_units) > 6:
            #    active_units_prune = var.sort(descending=True)[1][var.sort(descending=True)[0] > VAR_THRESHOLD][:top_dim].long()
            #    active_units = active_units.sort()[0]
            #   print('Active units prune: ' + ','.join(map(str, active_units_prune.tolist())))

            # Prune the uninformative latents
            qz_samples_norm = {a.item(): qz_samples_norm[:, a] for a in active_units}
            #qz_samples_norm = {a: qz_samples_norm[:,a] for a in range(10)}
            #print("qz samples norm size {}".format(qz_samples_norm.keys()))

            # compute the Gini empirical variance, vars is a dict
            #active_units = torch.arange(0,10)
            vars = gini_variance(qz_samples_norm, active_units)
            #print("Gini variance {}".format(vars.values()))

            # get the argmin
            d_star = min(vars, key=vars.get)
            #print("d_star {}".format(d_star))

            train_point = (d_star, k_dim)
            #print("train_point {}".format(train_point))

            training_set.append(train_point)

        # Matrix of votes

        #print("train set {}".format(training_set))

        vote = np.zeros((10, 5))
        for j in range(10):
            for k in range(5):
                vote[j, k] = sum([1 for d, k_f in training_set if d == j and k == k_f])

        """ vote =  array([[0., 0., 0., 0., 0.],
                           [0., 0., 1., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 4., 0., 0., 0.],
                           [0., 1., 3., 0., 0.],
                           [0., 0., 0., 1., 0.],
                           [2., 0., 0., 0., 0.],
                           [0., 0., 0., 3., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.]]) """

        # Compute the argmax, random if no count
        vote_list = []
        for row in vote:
            if sum(row) == 0:
                k_max = random.sample(range(5), 1)[0]
            else:
                k_max = np.argmax(row)
            vote_list.append(k_max)

        # vote list = [4, 2, 2, 1, 2, 3, 0, 3, 0, 1]
        #print("vote list {}".format(vote_list))
        #print(fdfdfdf)

        #len(vote_list) = 10, Vjk
        vote_list = np.array(vote_list).reshape(1,10)
        # append vote
        list_votes.append(vote_list)

    #convert to shape (votes, dims) -> (800, 10)
    list_votes = np.array(list_votes).reshape(-1,10)
    #print()
    #print("list votes shape {}".format(list_votes.shape))
    #print("list votes {}".format(list_votes))
    #print(fdfd)

    # put in a dict all the votes per dimension
    dict_cj = defaultdict(list)
    for j in range(3):
        for i in range(10):
            dict_cj[i].append(list_votes[j, i])

    """ defaultdict(list,
            {0: [4, 3, 4],
             1: [2, 2, 3],
             2: [2, 2, 3],
             3: [1, 1, 1],
             4: [2, 2, 2],
             5: [3, 3, 2],
             6: [0, 0, 0],
             7: [3, 3, 3],
             8: [0, 2, 0],
             9: [1, 1, 2]}) """

    # put in a dict the counts per dimension
    dict_votes_final = {}
    for key in dict_cj.keys():
        dict_votes_final[key] = Counter(dict_cj[key])

    """{0: Counter({3: 1, 4: 2}),
        1: Counter({2: 2, 3: 1}),
        2: Counter({2: 2, 3: 1}),
        3: Counter({1: 3}),
        4: Counter({2: 3}),
        5: Counter({2: 1, 3: 2}),
        6: Counter({0: 3}),
        7: Counter({3: 3}),
        8: Counter({0: 2, 2: 1}),
        9: Counter({1: 2, 2: 1})}"""

    # Get the metric values
    metric_values = []
    for i in range(10):
        j, count = dict_votes_final[i].most_common()[0]
        value = round(count / sum(dict_votes_final[i].values()), 3)
        metric_values.append(value)

    # Prune the values of the metric to 5.
    print("metric values {}".format(metric_values))
    factors_metric = [i for i in metric_values if i > 0.3][:6]
    print("factors values {}".format(metric_values))
    metric = sum(factors_metric) / 5

    train_model(train=True)

    return metric


def factor_vae_metric_shapes_v2(train_model, model, votes = 500):

    # I believe that one vote is for one seed. So this implementation
    # is just for 1 vote.

    train_model(train=False)

    L = 100
    M = 100
    K_big = 6
    D = model.z_dim
    training_set = []

    for k_dim in range(1, K_big):

        print("k_dim {}".format(k_dim))

        for i in range(M) :

            #print("vote {}, i {}".format(v,i))

            # fix one factor
            #sample a dimension
            #k_dim = random.sample(range(latents_sizes.shape[0]), 1)[0]
            # sample the factor k
            factor_k = random.sample(range(latents_sizes[k_dim]), 1)[0]

            #print("k_dim {}".format(k_dim))
            #print("factor_k {}".format(factor_k))

            ## Sample using the factor k
            latents_sampled = sample_latent(size=L)
            latents_sampled[:, k_dim] = factor_k
            indices_sampled = latent_to_index(latents_sampled)
            imgs_sampled = imgs[indices_sampled]
            # shape of imgs_sampled (L, 64, 64)

            xs = torch.from_numpy(imgs_sampled)
            #print("xs size {}".format(xs.size()))
            #print("xs type {}".format(xs.type()))

            xs = xs.type(torch.float)
            #print("xs type {}".format(xs.type()))

            xs_sampled = xs.unsqueeze(1).to("cuda")
            #xs_sampled = xs.unsqueeze(1).to("cpu")
            #print("xs sampled {}".format(xs_sampled.size()))

            # Sample Z
            qz_means = model(xs_sampled, decode=False)[1]
            qz_samples = model(xs_sampled, decode=False)[3]
            #print("qz_means size {}".format(qz_means.size()))

            qz_samples = qz_samples.cuda()
            #qz_samples = qz_samples.cpu()
            # qz_samples shape (L, D)

            # Compute the standard deviation
            s = torch.std(qz_samples, dim = 0)
            #print("s {}".format(s))
            #print("s size {} ".format(s.size()))

            # Normalise
            qz_samples_norm = qz_samples / s
            # print("qz samples norm size {}".format(qz_samples_norm.size()))
            # qz_samples_norm shape (100,10)

            # prune active units
            var = torch.std(qz_means.contiguous().view(L, D), dim=0).pow(2)  # pow is just **2, contiguous in memory
            active_units = torch.arange(0, D)[var > VAR_THRESHOLD].long()
            #print('Active units: ' + ','.join(map(str, active_units.tolist())))

            #if len(active_units) > 6:
            #    active_units_prune = var.sort(descending=True)[1][var.sort(descending=True)[0] > VAR_THRESHOLD][:top_dim].long()
            #    active_units = active_units.sort()[0]
            #   print('Active units prune: ' + ','.join(map(str, active_units_prune.tolist())))

            # Prune the uninformative latents
            qz_samples_norm = {a.item(): qz_samples_norm[:, a] for a in active_units}
            #qz_samples_norm = {a: qz_samples_norm[:,a] for a in range(10)}
            #print("qz samples norm size {}".format(qz_samples_norm.keys()))

            # compute the Gini empirical variance, vars is a dict
            #active_units = torch.arange(0,10)
            vars = gini_variance(qz_samples_norm, active_units)
            #print("Gini variance {}".format(vars.values()))

            # get the argmin
            d_star = min(vars, key=vars.get)
            #print("d_star {}".format(d_star))

            train_point = (d_star, k_dim)
            #print("train_point {}".format(train_point))

            training_set.append(train_point)

    # Matrix of votes
    #print("train set {}".format(training_set))

    vote = np.zeros((10, 6))
    for j in range(10):
        for k in range(1,6):
            vote[j, k] = sum([1 for d, k_f in training_set if d == j and k == k_f])

    """ vote =  array([[0., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 4., 0., 0., 0.],
                       [0., 1., 3., 0., 0.],
                       [0., 0., 0., 1., 0.],
                       [2., 0., 0., 0., 0.],
                       [0., 0., 0., 3., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.]]) """

    print("votes {}".format(vote))

    list_votes = vote

    final_votes_list = []
    counter = 0
    for row in list_votes:
        k_max = np.argmax(row)
        value_max = row[int(k_max)]
        if np.sum(row) > 0:
            factor_metric = value_max / np.sum(row)
        else:
            factor_metric = 0.0
        if sum(row) > 30.:
            final_votes_list.append(factor_metric)
            counter += 1
        else:
            final_votes_list.append(0.0)
    print("final_votes_list {}".format(final_votes_list))
    print("counter {}".format(counter))

    metric = np.sum(final_votes_list) / 6

    train_model(train=True)

    return metric



