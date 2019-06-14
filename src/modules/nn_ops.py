import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


def variable_init(m):
	if isinstance(m, nn.Linear):
		nn.init.xavier_normal_(m.weight, math.sqrt(2.))
		if m.bias is not None: nn.init.constant_(m.bias, 0.)

	elif isinstance(m, nn.BatchNorm1d):
		if m.weight is not None: nn.init.constant_(m.weight, 1.)
		if m.bias is not None: nn.init.constant_(m.bias, 0.)


def d_hinge_loss(d_real, d_fake, limit=1.):
    r = F.relu(limit - d_real).mean()
    f = F.relu(limit + d_fake).mean()
    return r + f


def g_hinge_loss(d_fake):
    loss = - d_fake.mean()
    return loss


def orth_regul(net, device='cuda'):
    orth_loss = torch.zeros(1, device=device)
    for name, param in net.named_parameters():
        if 'weight' in name:
            n_rows, n_cols = param.shape
            if n_cols >= n_rows:
                w_wT = param.mm(param.t())
                eye = torch.eye(n_rows, device=device)
                orth_loss += (w_wT - eye).norm(p=2)
    return orth_loss


def gradient_penalty(d_net, x_real, x_fake, a=None):
    assert x_real.size(0) == x_fake.size(0)
    n_samples = x_real.size(0)
    n_dims = len(x_real.size())
    
    epsilon = torch.rand(n_samples, *([1] * (n_dims - 1)), device=x_real.device)
    epsilon = epsilon.expand_as(x_real)

    x_hat = epsilon * x_real + ((1.0 - epsilon) * x_fake)
    if not x_hat.requires_grad: x_hat.requires_grad = True
    
    if a is not None: 
        out = d_net(x_hat, a)
    else: 
        out = d_net(x_hat)

    gradients = grad(
        [out],
        [x_hat],
        torch.ones(out.size(), device=x_hat.device),
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2.).mean()
    return penalty
				
