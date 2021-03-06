import os
import glob
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class BetaVAE(nn.Module):
    '''
      Creates a BetaVAE model with the same architecture as Higgins et al.

      - This only works for two imagesizes: 64x64 and 256x256.
    '''
    def __init__(self, latent_size=32, beta=10, imsize=256):
        super(BetaVAE, self).__init__()

        self.latent_size = latent_size
        self.beta = beta
        
        if imsize == 64:
            p=1
            op=0
            s=1
            self.n=1
        else:
            p= 0
            op=1
            s=2
            self.n=6

        # encoder
        self.encoder = nn.Sequential(
            self._conv(3, 32, p),
            self._conv(32, 32, p),
            self._conv(32, 64, p),
            self._conv(64, 64, p),
            self._conv(64, 256, stride=s),
        )
        self.fc_mu = nn.Linear(256*self.n*self.n, latent_size)
        self.fc_var = nn.Linear(256*self.n*self.n, latent_size)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(256, 64, stride=s),
            self._deconv(64, 64, padding=p),
            self._deconv(64, 32, padding=p),
            self._deconv(32, 32, padding=p, out_padding=op),
            self._deconv(32, 3, padding=p),
            nn.Sigmoid()
        )
        self.fc_z = nn.Linear(latent_size, 256*self.n*self.n)
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            if isinstance(self._modules[block], nn.Linear):
                kaiming_init(self._modules[block])
                continue
            for m in self._modules[block]:
                kaiming_init(m) 

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, 256*self.n*self.n)
        return self.fc_mu(x), self.fc_var(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 256, self.n, self.n)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def _conv(self, in_channels, out_channels, out_padding=0, stride=2, padding=0):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=stride, padding=out_padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def _deconv(self, in_channels, out_channels, out_padding=0, stride=2, padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=stride, output_padding=out_padding, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        recon_loss = F.mse_loss(recon_x, x, size_average=False).div(x.shape[0])

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        #kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_diverge = -0.5*(1+logvar-mu.pow(2) - logvar.exp())

        return recon_loss + self.beta * kl_diverge.sum(1).mean(0,True)

        #return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

