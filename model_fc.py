import torch.nn as nn

class VAE_MLP(nn.Module):
    def __init__(self, z_dim):
        super(VAE_MLP, self).__init__()

        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(4096, 1200),
            nn.ReLU(),
            nn.Linear(1200, 1200),
            nn.ReLU(),
            nn.Linear(1200, 2 * z_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def reparametrise(selfs, mu, logvar):

        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, decode = True):

        params = self.encoder(x)
        mu = params[:, :self.z_dim]
        logvar = params[:, self.z_dim:]
        z = self.reparametrise(mu, logvar)

        if decode:
            x_recon = self.decoder(z)
        else:
            x_recon = None

        return x_recon, mu, logvar, z

