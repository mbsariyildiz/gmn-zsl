import torch.nn as nn
from torch.nn import functional as F
from classifiers.base_compatibility import _BaseCompatibility


class Model(nn.Module):

    def __init__(self, d_in, d_out, n_hlayer, n_hunit):
        super().__init__()

        if n_hlayer == 1:
            self.layers = nn.Sequential(
                nn.Linear(d_in, n_hunit), nn.ReLU(),
                nn.Linear(n_hunit, d_out))
        elif n_hlayer == 2:
            self.layers = nn.Sequential(
                nn.Linear(d_in, n_hunit), nn.ReLU(),
                nn.Linear(n_hunit, n_hunit), nn.ReLU(),
                nn.Linear(n_hunit, d_out))
        else:
            raise NotImplementedError

    def forward(self, x, s):
        x = self.layers(x)
        x = F.linear(x, s)
        return x


class MultiLayerCompatibility(_BaseCompatibility):

    def __init__(self, d_ft, d_emb, args):
        super().__init__(d_ft, d_emb, args)
        self.net = Model(self.d_ft, self.d_emb, self.n_hlayer, self.n_hunit).to(self.device)
        self.reset()
