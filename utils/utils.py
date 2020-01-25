import torch
from torch.nn import functional as F
from torchvision.utils import save_image


class DataGather(object):
    def __init__(self, *args):
        self.keys = args
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return {arg:[] for arg in self.keys}

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


def traverse(train_model, model, test_imgs, save_path):

    train_model(train=False)

    # Decoding images
    x_recon, mu, logvar, z = model(test_imgs)
    x1 = F.sigmoid(x_recon).data
    canvas = torch.cat((test_imgs, x1), dim=1)
    num_colums = len(test_imgs)

    first_mu = mu[0]

    z_dim = len(first_mu)
    latents = torch.cat((first_mu, z[0, z_dim:]))

    trav_range = torch.linspace(-3, 3, num_colums)

    for i in range(z_dim):

        # put the same latents in order
        temp_latents = latents.repeat(num_colums, 1)
        # Instead of the latent, use the std dev
        temp_latents[:, i] = trav_range
        # retrieve the recons
        recons = model.decoder(temp_latents)
        recons1 = F.sigmoid(recons).data
        canvas = torch.cat((canvas, recons1), dim=1)

    img_size = canvas.shape[2:]
    canvas = canvas.transpose(0, 1).contiguous().view(-1, 1, *img_size)
    save_image(canvas, save_path, nrow=num_colums, pad_value=1)

    train_model(train=True)


def traverse_mlp(train_model, model, test_imgs, save_path):

    train_model(train=False)

    test_imgs_1 = test_imgs.unsqueeze(1)
    test_imgs_1 = test_imgs_1.view(-1, 4096)

    # Decoding images
    x_recon, mu, logvar, z = model(test_imgs_1)
    x1 = F.sigmoid(x_recon).data
    canvas = torch.cat((test_imgs, x1.view(-1, 1, 64, 64)), dim=1)
    num_colums = len(test_imgs)
    first_mu = mu[0]

    z_dim = len(first_mu)
    latents = torch.cat((first_mu, z[0, z_dim:]))

    trav_range = torch.linspace(-3, 3, num_colums)

    for i in range(z_dim):

        # put the same latents in order
        temp_latents = latents.repeat(num_colums, 1)
        # Instead of the latent, use the std dev
        temp_latents[:, i] = trav_range
        # retrieve the recons
        recons = model.decoder(temp_latents)
        recons1 = F.sigmoid(recons).data
        # SIZE (9, 1, 64, 64)

        canvas = torch.cat((canvas, recons1.view(-1, 1, 64, 64)), dim=1)

    img_size = canvas.shape[2:]
    canvas = canvas.transpose(0, 1).contiguous().view(-1, 1, *img_size)
    save_image(canvas, save_path, nrow=num_colums, pad_value=1)

    train_model(train=True)


def traverse_faces(train_model, model, test_imgs, save_path):

    train_model(train=False)

    # Decoding images

    x_recon, mu, logvar, z = model(test_imgs)
    x1 = F.sigmoid(x_recon).data
    x1_p = x1.unsqueeze(1)

    test_imgs_p = test_imgs.unsqueeze(1)
    canvas = torch.cat((test_imgs_p , x1_p), dim=1)
    num_colums = len(test_imgs)
    first_mu = mu[0]

    z_dim = len(first_mu)
    latents = torch.cat((first_mu, z[0, z_dim:]))
    trav_range = torch.linspace(-3, 3, num_colums)

    for i in range(z_dim):

        # put the same latents in order
        temp_latents = latents.repeat(num_colums, 1)
        # Instead of the latent, use the std dev
        temp_latents[:, i] = trav_range
        # retrieve the recons
        recons = model.decoder(temp_latents)
        recons1 = F.sigmoid(recons).data
        recons1_p = recons1.unsqueeze(1)
        canvas = torch.cat((canvas, recons1_p), dim=1)

    img_size = canvas.shape[2:]
    canvas = canvas.transpose(0, 1).contiguous().view(-1, *img_size)

    save_image(canvas, save_path, nrow=num_colums, pad_value=1)

    train_model(train=True)



