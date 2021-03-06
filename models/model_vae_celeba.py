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


class VAECelebA(nn.Module):
    def __init__(self, z_dim):
        super(VAECelebA, self).__init__()

        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * z_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 4 * 4),
            nn.ReLU(),
            Reshape(-1, 64, 4, 4),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1)
        )

    def reparametrise(self, mu, log_var):

        std = log_var.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, decode = True):

        params = self.encoder(x)
        mu = params[:, :self.z_dim]
        log_var = params[:, self.z_dim:]
        z = self.reparametrise(mu, log_var)

        if decode:
            x_recon = self.decoder(z)
        else:
            x_recon = None

        return x_recon, mu, log_var, z