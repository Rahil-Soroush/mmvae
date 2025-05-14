# Base VAE class definition

import torch
import torch.nn as nn

from utils import get_mean, kl_divergence
from vis import embed_umap, tensors_to_df


class VAE(nn.Module):
    def __init__(self, prior_dist, likelihood_dist, post_dist, enc, dec, params):
        super(VAE, self).__init__()
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = post_dist
        self.enc = enc
        self.dec = dec
        self.modelName = None
        self.params = params
        self._pz_params = None  # defined in subclass
        self._qz_x_params = None  # populated in `forward`
        self.llik_scaling = 1.0

    @property
    def pz_params(self):
        return self._pz_params

    @property
    def qz_x_params(self):
        if self._qz_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qz_x_params

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x, K=1):
        self._qz_x_params = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        # zs = qz_x.rsample(torch.Size([K]))
        # px_z = self.px_z(*self.dec(zs))
        #####
        zs = qz_x.rsample(torch.Size([K]))  # (K, B, D)
        zs = zs.view(-1, zs.size(-1))       # (K*B, D)
        mean, scale = self.dec(zs)          # (K*B, C, T)
        mean = mean.view(K, -1, *mean.shape[1:])   # (K, B, C, T)
        scale = scale.view(K, -1, *scale.shape[1:])  # same
        px_z = self.px_z(mean, scale)  # now px_z has batch_shape (K, B, C, T)
        #########
        return qz_x, px_z, zs

    # def generate(self, N, K):
    #     self.eval()
    #     with torch.no_grad():
    #         pz = self.pz(*self.pz_params)
    #         latents = pz.rsample(torch.Size([N]))
    #         px_z = self.px_z(*self.dec(latents))
    #         data = px_z.sample(torch.Size([K]))
    #     return data.view(-1, *data.size()[3:])
    
    ##########
    def generate(self, N, K):
        self.eval()
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([K, N]))  # (K, N, latent_dim)
            latents = latents.view(-1, latents.size(-1))  # (K*N, latent_dim)
            mean, scale = self.dec(latents)
            mean = mean.view(K, N, *mean.shape[1:])
            scale = scale.view(K, N, *scale.shape[1:])
            px_z = self.px_z(mean, scale)
            data = px_z.sample()  # (K, N, C, T)
        return data.view(-1, *data.shape[2:])  # (K*N, C, T)

    ########    
    # def reconstruct(self, data):
    #     self.eval()
    #     with torch.no_grad():
    #         qz_x = self.qz_x(*self.enc(data))
    #         latents = qz_x.rsample()  # no dim expansion
    #         px_z = self.px_z(*self.dec(latents))
    #         recon = get_mean(px_z)
    #     return recon
    
    ######
    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            latents = qz_x.rsample()  # (B, D)
            mean, scale = self.dec(latents)  # (B, C, T)
            px_z = self.px_z(mean, scale)
            recon = get_mean(px_z)  # shape: (B, C, T)
        return recon

    #########

    def analyse(self, data, K):
        self.eval()
        with torch.no_grad():
            qz_x, _, zs = self.forward(data, K=K)
            pz = self.pz(*self.pz_params)
            zss = [pz.sample(torch.Size([K, data.size(0)])).view(-1, pz.batch_shape[-1]),
                   zs.view(-1, zs.size(-1))]
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
            kls_df = tensors_to_df(
                [kl_divergence(qz_x, pz).cpu().numpy()],
                head='KL',
                keys=[r'KL$(q(z|x)\,||\,p(z))$'],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )
        return embed_umap(torch.cat(zss, 0).cpu().numpy()), \
            torch.cat(zsl, 0).cpu().numpy(), \
            kls_df
