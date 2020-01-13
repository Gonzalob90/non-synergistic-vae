import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Discriminator(nn.Module):
    def __init__(self, num_latents):
        super(Discriminator, self).__init__()

        self.num_latents = num_latents

        self.net = nn.Sequential(
            nn.Linear(num_latents, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 2)
            )

    def forward(self, z):
        return self.net(z).squeeze()


class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()

        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            #nn.ReLU(),
            nn.Linear(128, 2 * z_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 4 * 4),
            nn.ReLU(),
            Reshape(-1, 64, 4, 4),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1)
        )

    def reparametrise(self, mu, log_var):

        std = log_var.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, decode=True):

        params = self.encoder(x)
        mu = params[:, :self.z_dim]
        log_var = params[:, self.z_dim:]
        z = self.reparametrise(mu, log_var)

        if decode:
            x_recon = self.decoder(z)
        else:
            x_recon = None

        return x_recon, mu, log_var, z
