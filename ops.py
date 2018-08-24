import torch
import torch.nn.functional as F


def recon_loss(x, x_recon):
    n = x.size(0)
    loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(n)
    return loss


def kl_div(mu, logvar):
    kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
    return kld

def kl_div_syn(mu, logvar):
    kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
    return kld

def kl_div_uni_dim(mu, logvar):
    kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
    return kld

def kl_div_mean(mu, logvar):
    kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(1).mean()
    return kld

def kl_div_mean_syn_1A(mu, logvar):
    kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
    #print(kld.size())
    return kld

def permute_dims(z):
    # Test for two dim, B X d
    assert z.dim() == 2

    # get the B
    B, _ = z.size()
    perm_z = []
    # take 1 element in columns
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)

