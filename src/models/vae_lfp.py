# LFP model specification (Time-Series Inspired by Image & Sentence VAE)

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import Constants
from vis import plot_embeddings, plot_kls_df
from .vae import VAE


# Encoder: Time-series to latent parameters
class Enc(nn.Module):
    def __init__(self, input_channels, input_length, latent_dim):
        super(Enc, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # output: (N, 64, 1)
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):  # x: (N, C, T)
        x = self.conv(x).squeeze(-1)  # (N, 64)
        mu = self.fc_mu(x)
        logvar = F.softplus(self.fc_logvar(x)) + Constants.eta
        return mu, logvar


# Decoder: latent → time-series
# class Dec(nn.Module):
#     def __init__(self, output_channels, output_length, latent_dim):
#         super(Dec, self).__init__()
#         self.output_length = output_length
#         self.deconv_input = nn.Linear(latent_dim, 64 * (output_length // 4))
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.ConvTranspose1d(32, output_channels, kernel_size=4, stride=2, padding=1),
#             nn.Tanh()  # instead of Sigmoid
#         )
#         # self.deconv = nn.Sequential(
#         #     nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
#         #     nn.BatchNorm1d(32),
#         #     nn.ReLU(),
#         #     nn.ConvTranspose1d(32, output_channels, kernel_size=4, stride=2, padding=1),
#         #     nn.Sigmoid()
#         # )

#     def forward(self, z):  # z: (N, latent_dim)
#         x = self.deconv_input(z).view(z.size(0), 64, -1)  # (N, 64, L)
#         # print("decoder input reshaped to:", x.shape)
#         out = self.deconv(x)  # (N, C, T)
#         # print("decoder output shape:", out.shape)
#         # return out.clamp(Constants.eta, 1 - Constants.eta), torch.tensor(0.75).to(z.device)
#         return out, torch.tensor(0.75).to(z.device)

class Dec(nn.Module):
    """Decoder: latent → time-series (N, latent_dim) → (N, C, T)"""
    def __init__(self, output_channels, output_length, latent_dim):
        super(Dec, self).__init__()
        self.output_length = output_length
        self.deconv_input = nn.Linear(latent_dim, 64 * (output_length // 4))  # e.g. 64 * 61
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),  # -> (N, 32, ~122)
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(32, output_channels, kernel_size=5, stride=2, padding=2, output_padding=1),  # -> (N, C, 244)
            nn.Tanh()  # for real-valued LFPs
        )

    def forward(self, z):  # z: (N, latent_dim)
        x = self.deconv_input(z).view(z.size(0), 64, -1)  # (N, 64, L)
        out = self.deconv(x)  # (N, C, T)
        return out, torch.tensor(0.75).to(z.device)

# VAE class for LFP
class LFPVAE(VAE):
    def __init__(self, params):
        enc = Enc(params.input_channels, params.input_length, params.latent_dim)
        dec = Dec(params.input_channels, params.input_length, params.latent_dim)
        super().__init__(
            dist.Laplace,
            dist.Laplace,
            dist.Laplace,
            enc,
            dec,
            params
        )
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)
        ])
        self.modelName = getattr(params, 'name', 'lfp')
        self.dataSize = (params.input_channels, params.input_length)
        self.llik_scaling = 1.

    @property
    def pz_params(self):
        return self._pz_params[0], F.softplus(self._pz_params[1]) + Constants.eta

    def generate(self, runPath, epoch):
        N, K = 64, 4
        samples = super(LFPVAE, self).generate(N, K).cpu()
        samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)  # N x K x C x T
        torch.save(samples, f"{runPath}/gen_samples_{epoch:03d}.pt")

    def reconstruct(self, data, runPath=None, epoch=0):
        recon = super(LFPVAE, self).reconstruct(data)
        if runPath is not None:
            torch.save((data.cpu(), recon.cpu()), f"{runPath}/recon_{epoch:03d}.pt")
        return recon

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(LFPVAE, self).analyse(data, K=10)
        labels = ['Prior', self.modelName.lower()]
        plot_embeddings(zemb, zsl, labels, f"{runPath}/emb_umap_{epoch:03d}.png")
        plot_kls_df(kls_df, f"{runPath}/kl_distance_{epoch:03d}.png")
