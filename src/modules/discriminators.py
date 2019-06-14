import torch
import torch.nn as nn
from .nn_ops import variable_init


class ConditionalDiscriminator(nn.Module):

    def __init__(self,
                 n_attr,
                 n_hlayer,
                 n_hunit,
                 normalize_infeature=False,
                 dropout=0.0,
                 leakiness=0.2):
        super().__init__()
        self.n_hunit = n_hunit
        self.n_attr = n_attr
        self.n_hlayer = n_hlayer
        self.normalize_infeature = normalize_infeature
        self.dropout = dropout
        self.leakiness = leakiness

        if self.n_hlayer > 0:
            self.main = nn.ModuleList([
                nn.Linear(2048 + self.n_attr, self.n_hunit),
                nn.LeakyReLU(self.leakiness, inplace=True)])
            for _ in range(1, self.n_hlayer):
                self.main.extend([
                    nn.Linear(self.n_hunit, self.n_hunit),
                    nn.LeakyReLU(leakiness, inplace=True)])
            self.main.append(
                nn.Linear(self.n_hunit, 1))

        else:
            self.main = nn.Linear(2048, 1)

        self.apply(variable_init)
            
    def forward(self, x, a):
        if self.normalize_infeature:
            x = nn.functional.normalize(x, dim=1, p=2)
        if self.training and self.dropout > 0.:
            mask = torch.bernoulli(torch.ones_like(x) * (1.0 - self.dropout))
            x = x * mask

        out = torch.cat([x, a], dim=1)
        for lix in range(len(self.main)):
            out = self.main[lix](out)
        return out.squeeze(1)

    def extra_repr(self):
        extra_str = 'n_attr={n_attr}, ' \
                    'n_hlayer={n_hlayer}, ' \
                    'n_hunit={n_hunit}, ' \
                    'dropout={dropout}, ' \
                    'leakiness={leakiness}, ' \
                    'normalize_infeature={normalize_infeature}'.format(**self.__dict__)
        return extra_str
