import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.autograd import Variable
import brewer2mpl
bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
colors = bmap.mpl_colors

plt.style.use('ggplot')

VAR_THRESHOLD = 1e-2

init_seed = 120
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)

def plot_gt_shapes(train_model, model, dataloder_gt, save_path):

    train_model(train=False)

    N = len(dataloder_gt.dataset)  # number of data samples
    K = model.z_dim
    #nparams = 1

    # print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K)
    #print(qz_params.size())

    n = 0
    for xs, xs2 in dataloder_gt:
        batch_size = xs.size(0)
        #xs = Variable(xs.view(batch_size, 1, 64, 64).cuda(), volatile=True)
        #xs = Variable(xs.view(batch_size, 1, 64, 64).cpu())#  only inference
        xs = Variable(xs.view(batch_size, 1, 64, 64).cuda())
        #print(model(xs, decode=False)[1].size())
        #print(qz_params[n:n + batch_size].size())
        with torch.no_grad():
            qz_params[n:n + batch_size] = model(xs, decode=False)[1]
        n += batch_size

    # Primitive factors: shapes, scale, orientation, position, position
    #qz_params = qz_params.view(3, 6, 40, 32, 32, K, nparams)
    qz_params = qz_params.view(3, 6, 40, 32, 32, K)

    # z_j is inactive if Var_x(E[z_j|x]) < eps.
    #qz_means = qz_params[:, :, :, :, :, :, 0]
    qz_means = qz_params

    var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2) # pow is just **2, contiguous in memory
    active_units = torch.arange(0, K)[var > VAR_THRESHOLD].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))

    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, model.z_dim))

    z_inds = active_units

    # subplots where subplot[i, j] is gt_i vs. z_j
    mean_scale = qz_means.mean(2).mean(2).mean(2)  # (shape, scale, latent)
    mean_rotation = qz_means.mean(1).mean(2).mean(2)  # (shape, rotation, latent)
    mean_pos = qz_means.mean(0).mean(0).mean(0)  # (pos_x, pos_y, latent)

    #plt.rc('axes', labelsize=8)

    fig = plt.figure(figsize=(3, len(z_inds)))  # default is (8,6)
    gs = gridspec.GridSpec(len(z_inds), 3)
    gs.update(wspace=0, hspace=0)  # set the spacing between axes.

    # GRAPHS
    vmin_pos = torch.min(mean_pos)
    vmax_pos = torch.max(mean_pos)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[i * 3])
        ax.imshow(mean_pos[:, :, j].numpy(), cmap=plt.get_cmap('coolwarm'), vmin=vmin_pos, vmax=vmax_pos)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(r'$z_' + str(j.item()) + r'$')
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'pos')

    vmin_scale = torch.min(mean_scale)
    vmax_scale = torch.max(mean_scale)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[1 + i * 3])
        ax.plot(mean_scale[0, :, j].numpy(), color=colors[2])
        ax.plot(mean_scale[1, :, j].numpy(), color=colors[0])
        ax.plot(mean_scale[2, :, j].numpy(), color=colors[1])
        ax.set_ylim([vmin_scale, vmax_scale])
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'scale')

    vmin_rotation = torch.min(mean_rotation)
    vmax_rotation = torch.max(mean_rotation)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[2 + i * 3])
        ax.plot(mean_rotation[0, :, j].numpy(), color=colors[2])
        ax.plot(mean_rotation[1, :, j].numpy(), color=colors[0])
        ax.plot(mean_rotation[2, :, j].numpy(), color=colors[1])
        ax.set_ylim([vmin_rotation, vmax_rotation])
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'rotation')

    fig.text(0.5, 0.03, 'Ground Truth', ha='center')
    fig.text(0.01, 0.5, 'Learned Latent Variables ', va='center', rotation='vertical')
    plt.savefig(save_path)
    plt.close()

    train_model(train=True)


def plot_gt_shapes_bvae(train_model, model, dataloder_gt, save_path):

    train_model(train=False)

    N = len(dataloder_gt.dataset)  # number of data samples
    K = model.z_dim
    #nparams = 1

    # print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K)
    #print(qz_params.size())

    n = 0
    for xs, xs2 in dataloder_gt:
        batch_size = xs.size(0)
        #xs = Variable(xs.view(batch_size, 1, 64, 64).cuda(), volatile=True)
        #xs = Variable(xs.view(batch_size, 1, 64, 64).cpu())#  only inference
        #xs = Variable(xs.view(batch_size, 1, 64, 64).cuda())
        xs = Variable(xs.view(batch_size, 4096).cuda())
        #print(model(xs, decode=False)[1].size())
        #print(qz_params[n:n + batch_size].size())
        with torch.no_grad():
            qz_params[n:n + batch_size] = model(xs, decode=False)[1]
        n += batch_size

    # Primitive factors: shapes, scale, orientation, position, position
    #qz_params = qz_params.view(3, 6, 40, 32, 32, K, nparams)
    qz_params = qz_params.view(3, 6, 40, 32, 32, K)

    # z_j is inactive if Var_x(E[z_j|x]) < eps.
    #qz_means = qz_params[:, :, :, :, :, :, 0]
    qz_means = qz_params

    var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2) # pow is just **2, contiguous in memory
    active_units = torch.arange(0, K)[var > VAR_THRESHOLD].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))

    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, model.z_dim))

    z_inds = active_units

    # subplots where subplot[i, j] is gt_i vs. z_j
    mean_scale = qz_means.mean(2).mean(2).mean(2)  # (shape, scale, latent)
    mean_rotation = qz_means.mean(1).mean(2).mean(2)  # (shape, rotation, latent)
    mean_pos = qz_means.mean(0).mean(0).mean(0)  # (pos_x, pos_y, latent)

    #plt.rc('axes', labelsize=8)

    fig = plt.figure(figsize=(3, len(z_inds)))  # default is (8,6)
    gs = gridspec.GridSpec(len(z_inds), 3)
    gs.update(wspace=0, hspace=0)  # set the spacing between axes.

    # GRAPHS
    vmin_pos = torch.min(mean_pos)
    vmax_pos = torch.max(mean_pos)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[i * 3])
        ax.imshow(mean_pos[:, :, j].numpy(), cmap=plt.get_cmap('coolwarm'), vmin=vmin_pos, vmax=vmax_pos)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(r'$z_' + str(j.item()) + r'$')
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'pos')

    vmin_scale = torch.min(mean_scale)
    vmax_scale = torch.max(mean_scale)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[1 + i * 3])
        ax.plot(mean_scale[0, :, j].numpy(), color=colors[2])
        ax.plot(mean_scale[1, :, j].numpy(), color=colors[0])
        ax.plot(mean_scale[2, :, j].numpy(), color=colors[1])
        ax.set_ylim([vmin_scale, vmax_scale])
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'scale')

    vmin_rotation = torch.min(mean_rotation)
    vmax_rotation = torch.max(mean_rotation)
    for i, j in enumerate(z_inds):
        ax = fig.add_subplot(gs[2 + i * 3])
        ax.plot(mean_rotation[0, :, j].numpy(), color=colors[2])
        ax.plot(mean_rotation[1, :, j].numpy(), color=colors[0])
        ax.plot(mean_rotation[2, :, j].numpy(), color=colors[1])
        ax.set_ylim([vmin_rotation, vmax_rotation])
        ax.set_xticks([])
        ax.set_yticks([])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        if i == len(z_inds) - 1:
            ax.set_xlabel(r'rotation')

    fig.text(0.5, 0.03, 'Ground Truth', ha='center')
    fig.text(0.01, 0.5, 'Learned Latent Variables ', va='center', rotation='vertical')
    plt.savefig(save_path)
    plt.close()

    train_model(train=True)