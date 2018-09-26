
import torch
import numpy as np
import math
from torch.autograd import Variable

metric_name = "MIG"

# tensor.select(2,index) tensor[:,:,index]

def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))


def MIG(mi_normed):
    return torch.mean(mi_normed[:, 0] - mi_normed[:, 1])


def compute_metric_shapes(marginal_entropies, cond_entropies):
    factor_entropies = [6, 40, 32, 32]

    # None put an exta dimension at the beginning
    mutual_infos = marginal_entropies[None] - cond_entropies

    # Choose the sorted not the index and truncate using min=0
    mutual_infos = torch.sort(mutual_infos, dim=1, descending=True)[0].clamp(min=0)

    # Take the logs and put an extra dimension of 1 in dim=1
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    metric = eval(metric_name)(mi_normed)
    return metric


##########

def log_density(sample, mu, logsigma):

    c = torch.Tensor([np.log(2 * np.pi)]).type_as(sample.data)
    inv_sigma = torch.exp(-logsigma)
    tmp = (sample - mu) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * logsigma + c)


def estimate_entropies1(qz_samples, qz_mu, qz_logvar, n_samples=10000, weights=None):
    """Computes the term:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    and
        E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    where q(z) = 1/N sum_n=1^N q(z|x_n).
    Assumes samples are from q(z|x) for *all* x in the dataset.
    Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).
    Computes numerically stable NLL:
        - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)
    Inputs:
    -------
        qz_samples (K, N) Variable
        qz_params  (N, K, nparams) Variable
        weights (N) Variable
    """

    # Only take a sample subset of the samples
    # index select indexes along dimension 1, qz_samples (K, N): K is the dimension of z, N is the samples
    # qz samples size(1) = N, basically gets the 10k samples of (K, n_samples)
    qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:n_samples].cuda()))

    # S is num_samples
    # qz_params size (N,K,nparams)
    K, S = qz_samples.size()
    N, _ = qz_mu.size()

    assert(K == qz_mu.size(1))

    weights = -math.log(N)
    entropies = torch.zeros(K).cuda()

    # Iterate over all the 10k samples
    k = 0
    while k < S:
        batch_size = min(10, S - k)
        logqz_i = log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_mu.view(N, K, 1).expand(N, K, S)[:, :, k:k + batch_size],
            qz_logvar.view(N, K, 1).expand(N, K, S)[:, :, k:k + batch_size])
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        entropies += - logsumexp(logqz_i + weights, dim=0, keepdim=False).data.sum(1)

    entropies /= S
    return entropies


def sample1(mu, logsigma):

    std_z = torch.randn(mu.size()).type_as(mu.data)
    sample = std_z * torch.exp(logsigma) + mu
    return sample


def mutual_info_metric_shapes(train_model, model, dataloder_gt):

    train_model(train=False)

    N = len(dataloder_gt.dataset)  # number of data samples
    K = model.z_dim

    print('Computing q(z|x) distributions.')
    qz_mu = torch.Tensor(N, K)
    qz_logvar = torch.Tensor(N, K)

    n = 0
    for xs, xs2 in dataloder_gt:
        batch_size = xs.size(0)
        # xs = Variable(xs.view(batch_size, 1, 64, 64).cuda(), volatile=True)
        # xs = Variable(xs.view(batch_size, 1, 64, 64).cpu())#  only inference
        xs = Variable(xs.view(batch_size, 1, 64, 64).cuda())
        # print(model(xs, decode=False)[1].size())
        # print(qz_params[n:n + batch_size].size())
        with torch.no_grad():
            qz_mu[n:n + batch_size] = model(xs, decode=False)[1]
            qz_logvar[n:n + batch_size] = model(xs, decode=False)[2]
        n += batch_size

    qz_mu = qz_mu.view(3, 6, 40, 32, 32, K).cuda()
    qz_logvar = qz_logvar.view(3, 6, 40, 32, 32, K).cuda()
    qz_samples = sample1(qz_mu, qz_logvar)

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies1(
        qz_samples.view(N, K).transpose(0, 1),
        qz_mu.view(N, K),
        qz_logvar.view(N, K))

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(4, K)


    print('Estimating conditional entropies for scale.')
    for i in range(6):
        qz_samples_scale = qz_samples[:, i, :, :, :, :].contiguous()
        qz_mu_scale = qz_mu[:, i, :, :, :, :].contiguous()
        qz_logvar_scale = qz_logvar[:, i, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies1(
            qz_samples_scale.view(N // 6, K).transpose(0, 1),
            qz_mu_scale.view(N // 6, K),
            qz_logvar_scale.view(N // 6, K))

        cond_entropies[0] += cond_entropies_i.cpu() / 6


    print('Estimating conditional entropies for orientation.')
    for i in range(40):
        qz_samples_scale = qz_samples[:, :, i, :, :, :].contiguous()
        qz_mu_scale = qz_mu[:, :, i, :, :, :].contiguous()
        qz_logvar_scale = qz_logvar[:, :, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies1(
            qz_samples_scale.view(N // 40, K).transpose(0, 1),
            qz_mu_scale.view(N // 40, K),
            qz_logvar_scale.view(N // 40, K))

        cond_entropies[1] += cond_entropies_i.cpu() / 40


    print('Estimating conditional entropies for pos x.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, i, :, :].contiguous()
        qz_mu_scale = qz_mu[:, :, :, i, :, :].contiguous()
        qz_logvar_scale = qz_logvar[:, :, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies1(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_mu_scale.view(N // 32, K),
            qz_logvar_scale.view(N // 32, K))

        cond_entropies[2] += cond_entropies_i.cpu() / 32


    print('Estimating conditional entropies for pox y.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, :, i, :].contiguous()
        qz_mu_scale = qz_mu[:, :, :, :, i, :].contiguous()
        qz_logvar_scale = qz_logvar[:, :, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies1(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_mu_scale.view(N // 32, K),
            qz_logvar_scale.view(N // 32, K))

        cond_entropies[3] += cond_entropies_i.cpu() / 32


    metric = compute_metric_shapes(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies