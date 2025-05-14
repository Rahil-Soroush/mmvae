# GPi-STN using shared LFPVAE class for both modalities
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from .mmvae import MMVAE
from .vae_lfp import LFPVAE

class GPI_STN(MMVAE):
    def __init__(self, params):
        # Instantiate two LFPVAE classes with different names but shared architecture
        def make_lfpvae(name, channels):
            sub_params = params.__dict__.copy()
            sub_params['name'] = name
            sub_params['input_channels'] = channels
            sub_params['input_length'] = params.input_length
            sub_params['latent_dim'] = params.latent_dim
            sub_params['num_hidden_layers'] = params.num_hidden_layers
            sub_params['learn_prior'] = params.learn_prior
            class Dummy: pass
            return LFPVAE(type('Params', (), sub_params))

        super().__init__(
            dist.Laplace,
            params,
            lambda p: make_lfpvae("gpi", params.gpi_channels),
            lambda p: make_lfpvae("stn", params.stn_channels)
        )

        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)
        ])
        self.modelName = 'gpi-stn'

    @property
    def pz_params(self):
        return self._pz_params[0], F.softplus(self._pz_params[1]) + 1e-6

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        data_save_dir = "../data/lfp_segments"
        subj = getattr(self, 'subj', 'subj1')
        save_dir = os.path.join(data_save_dir, subj)

        gpi_train = torch.load(os.path.join(save_dir, "gpi_train_off.pt")).permute(0, 2, 1)
        gpi_test = torch.load(os.path.join(save_dir, "gpi_test_off.pt")).permute(0, 2, 1)
        stn_train = torch.load(os.path.join(save_dir, "stn_train_off.pt")).permute(0, 2, 1)
        stn_test = torch.load(os.path.join(save_dir, "stn_test_off.pt")).permute(0, 2, 1)

        train_dataset = TensorDataset(gpi_train, stn_train)
        test_dataset = TensorDataset(gpi_test, stn_test)

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train_loader, test_loader

    def generate(self, runPath, epoch):
        N = 16
        samples_list = super().generate(N)
        for i, samples in enumerate(samples_list):
            torch.save(samples.cpu(), f"{runPath}/gen_samples_{i}_{epoch:03d}.pt")

    def reconstruct(self, data, runPath, epoch):
        recons_out = []
        recons_mat = super().reconstruct([d[:8] for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                _data = data[r][:8].cpu()
                recon = recon.squeeze(0).cpu()
                torch.save((_data, recon), f"{runPath}/recon_{r}x{o}_{epoch:03d}.pt")
                recons_out.append((_data, recon))
        return recons_out

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super().analyse(data, K=10)
        pd.DataFrame(kls_df).to_csv(f"{runPath}/kl_{epoch:03d}.csv")
        torch.save({'zemb': zemb, 'zsl': zsl}, f"{runPath}/embeddings_{epoch:03d}.pt")
