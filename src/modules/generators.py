import torch
import torch.nn as nn
from .nn_ops import variable_init


def get_generator(gen_type):
    if gen_type == 'latent_noise':
        return GeneratorLN
    elif gen_type == 'attribute_concat':
        return GeneratorAC


class GeneratorLN(nn.Module):

    def __init__(self,
                 n_attr,
                 d_noise,
                 n_hlayer,
                 n_hunit,
                 normalize_noise=0,
                 dropout=0.,
                 leakiness=0.2):
        super().__init__()
        self.n_hunit = n_hunit
        self.n_attr = n_attr
        self.d_noise = d_noise
        self.n_hlayer = n_hlayer
        self.normalize_noise = normalize_noise
        self.dropout = dropout
        self.leakiness = leakiness

        self.noise_mu = nn.Linear(self.n_attr, self.d_noise)
        self.noise_logvar = nn.Linear(self.n_attr, self.d_noise)

        if self.n_hlayer > 0:
            self.main = nn.ModuleList([
                nn.Linear(self.d_noise, self.n_hunit ),
                nn.LeakyReLU(leakiness, inplace=True),
                nn.Dropout(p=dropout)])
            for _ in range(1, self.n_hlayer):
                self.main.extend([
                    nn.Linear(self.n_hunit, self.n_hunit),
                    nn.LeakyReLU(leakiness, inplace=True),
                    nn.Dropout(p=dropout)])
            self.main.extend([
                nn.Linear(self.n_hunit, 2048), nn.ReLU(inplace=True)])

        else:
            self.main = nn.Sequential(
                nn.Linear(self.d_noise, 2048),
                nn.ReLU(inplace=True))

        self.apply(variable_init)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, a):
        mu = self.noise_mu(a)
        logvar = self.noise_logvar(a)
        z = self.reparameterize(mu, logvar)

        if self.normalize_noise:
            z = nn.functional.normalize(z, dim=1, p=2)

        if self.training and self.dropout > 0.:
            mask = torch.bernoulli(torch.ones_like(z) * (1.0 - self.dropout))
            z = z * mask

        out = z
        for lix in range(len(self.main)):
            out = self.main[lix](out)

        return out

    def extra_repr(self):
        extra_str = 'd_noise={d_noise}, ' \
                    'n_attr={n_attr}, ' \
                    'n_hlayer={n_hlayer}, ' \
                    'n_hunit={n_hunit}, ' \
                    'dropout={dropout}, ' \
                    'leakiness={leakiness}, ' \
                    'normalize_noise={normalize_noise}'.format(**self.__dict__)
        return extra_str


class GeneratorAC(nn.Module):

    def __init__(self,
                 n_attr,
                 d_noise,
                 n_hlayer,
                 n_hunit,
                 normalize_noise=0,
                 dropout=0.,
                 leakiness=0.2):
        super().__init__()
        self.n_hunit = n_hunit
        self.n_attr = n_attr
        self.d_noise = d_noise
        self.n_hlayer = n_hlayer
        self.normalize_noise = normalize_noise
        self.dropout = dropout
        self.leakiness = leakiness

        if self.n_hlayer > 0:
            self.main = nn.ModuleList([
                nn.Linear(
                    self.n_attr + self.d_noise, self.n_hunit),
                    nn.LeakyReLU(leakiness, inplace=True),
                    nn.Dropout(p=dropout)])
            for _ in range(1, self.n_hlayer):
                self.main.extend([
                    nn.Linear(self.n_hunit, self.n_hunit),
                    nn.LeakyReLU(leakiness, inplace=True),
                    nn.Dropout(p=dropout)])
            self.main.extend([
                nn.Linear(self.n_hunit, 2048), nn.ReLU(inplace=True)])

        else:
            self.main = nn.Sequential(
                nn.Linear(self.n_attr + self.d_noise, 2048),
                nn.ReLU(inplace=True))

        self.apply(variable_init)

    def forward(self, a):
        n_sample = a.size(0)
        z = torch.randn(n_sample, self.d_noise, device=a.device)
        if self.normalize_noise:
            z = nn.functional.normalize(z, dim=1, p=2)

        out = torch.cat([z, a], dim=1)
        for lix in range(len(self.main)):
            out = self.main[lix](out)
        return out

    def extra_repr(self):
        extra_str = 'd_noise={d_noise}, ' \
                    'n_attr={n_attr}, ' \
                    'n_hlayer={n_hlayer}, ' \
                    'n_hunit={n_hunit}, ' \
                    'dropout={dropout}, ' \
                    'leakiness={leakiness}, ' \
                    'normalize_noise={normalize_noise}'.format(**self.__dict__)
        return extra_str
