import numpy as np
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
    #print("x_recon {}".format(x_recon))
    x1 = F.sigmoid(x_recon).data
    #print("x1 {}".format(x1))
    #print("z {}".format(z.size()))
    canvas = torch.cat((test_imgs, x1), dim=1)
    #print("canvas size {}".format(canvas.size()))
    num_colums = len(test_imgs)
    #print("num columns {}".format(num_colums))

    #print("mu size {}".format(mu.size()))
    first_mu = mu[0]
    #print("first_mu {}".format(first_mu))

    z_dim = len(first_mu)
    #print("z_dim {}".format(z_dim))
    #print("z [0, zdim:] {}".format(z[0, z_dim:].size()))
    latents = torch.cat((first_mu, z[0, z_dim:]))
    #print("latents {}".format(latents))
    #print("latents size {}".format(latents.size()))

    trav_range = torch.linspace(-3, 3, num_colums)
    #print("trav_range {}".format(trav_range))

    for i in range(z_dim):

        #print("dim latent {}".format(i))
        # put the same latents in order
        temp_latents = latents.repeat(num_colums, 1)
        #print("temp latents size {}".format(temp_latents.size()))
        #print("temp latents {}".format(temp_latents))
        # Instead of the latent, use the std dev
        temp_latents[:, i] = trav_range
        #print("temp latents {}".format(temp_latents))
        # retrieve the recons
        recons = model.decoder(temp_latents)
        recons1 = F.sigmoid(recons).data
        #print("recons{}".format(recons[1,0,:5,:5]))
        #print("canvas before {}".format(canvas[1,:,:5,:5]))
        canvas = torch.cat((canvas, recons1), dim=1)
        #print("canvas after {}".format(canvas[1, :, :5, :5]))

        #print("canvas size {}".format(canvas.size()))

    img_size = canvas.shape[2:]
    #print("img size {}".format(img_size))
    canvas = canvas.transpose(0, 1).contiguous().view(-1, 1, *img_size)
   # print(canvas.size)
    save_image(canvas, save_path, nrow=num_colums, pad_value=1)

    train_model(train=True)



