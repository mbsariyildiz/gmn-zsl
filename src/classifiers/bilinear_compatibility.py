import torch.nn as nn
from torch.nn import functional as F
from classifiers.base_compatibility import _BaseCompatibility


class Model(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.layer = nn.Linear(d_in, d_out, True)

    def forward(self, x, s):
        x = self.layer(x)
        x = F.linear(x, s)
        return x
        


class BilinearCompatibility(_BaseCompatibility):

    def __init__(self, d_ft, d_emb, args):
        super().__init__(d_ft, d_emb, args)
        self.net = Model(self.d_ft, self.d_emb).to(self.device)
        self.reset()

