import numpy as np
import torch as th
from torch import nn
from torch import optim
from sklearn.metrics import confusion_matrix
import utils
from data import index_labels
from modules.nn_ops import variable_init


class Network(nn.Module):

    def __init__(self, d_ft, d_attr):
        super().__init__()
        self.layer = nn.Linear(d_ft, d_attr)
        self.apply(variable_init)

    def forward(self, x):
        x = self.layer(x)
        return x


class Regressor(object):

    def __init__(self, args, d_ft, d_attr):
        self.args = args
        self.lr = args.lr if 'lr' in args else 0.0
        self.wd = args.wd if 'wd' in args else 0.0
        self.device = args.device if 'device' in args else 'cpu'

        self.net = Network(d_ft, d_attr).to(self.device)
        self.optim = optim.Adam(
            self.net.parameters(), self.lr, weight_decay=self.wd)
        self.criterion = nn.MSELoss().to(self.device)

    def _compute_logits(self, s_hat, S):
        # Compute distance between each element in s_hat
        # and each element in S. Note that this function
        # returns the negative logits so that we can still
        # take argmax.
        n = s_hat.size(0)
        d = s_hat.size(1)
        m = S.size(0)
        with th.no_grad():
            logits = nn.functional.pairwise_distance(
                s_hat[:, None, :].expand(n, m, d).reshape((-1, d)),
                S[None, :, :].expand(n, m, d).reshape((-1, d))
            ).reshape((n, m))

        return -logits

    def loss(self, x, s):
        s_hat = self.net(x)
        loss = self.criterion(s_hat, s)
        return loss

    def train_step(self, x, y, S):
        s = S[y]
        s_hat = self.net(x)
        loss = self.criterion(s_hat, s)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        logits = self._compute_logits(s_hat, S)
        prec1 = utils.accuracy(logits, y)[0]
        return prec1.item(), loss.item()

    def train_epoch(self, iterator, S):
        train_loss = utils.AverageMeter()
        train_acc = utils.AverageMeter()

        self.net.train()

        for x, y in iterator:
            prec1, loss = self.train_step(x, y, S)
            train_acc.update(prec1, x.size(0))
            train_loss.update(loss, x.size(0))

        return train_loss.avg, train_acc.avg

    def test(self, iterator, S, C=None, confmat_path=''):
        labels = th.zeros(iterator.n_sample, dtype=th.int64, device=self.device)
        predictions = th.zeros(iterator.n_sample, dtype=th.int64, device=self.device)

        s_ix = 0
        self.net.eval()

        with th.no_grad():

            for x, y in iterator:
                bs = x.size(0)

                s_hat = self.net(x)
                logits = self._compute_logits(s_hat, S)
                if C is not None:
                    # take the logits of the classes in C
                    # (for ZSL evaluation,)
                    logits = logits[:, C]
                _, preds = logits.max(dim=1)
                labels[s_ix:s_ix+bs] = y
                predictions[s_ix:s_ix+bs] = preds
                s_ix += bs

            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()
            if C is not None:
                # map labels to be in [0, C.shape[0]]
                labels = index_labels(labels, C.cpu().numpy())

        acc = utils.normalized_acc(labels, predictions)

        if confmat_path:

            classes = np.arange(S.size(0))
            if C is not None:
                classes = np.arange(C.size(0))
            cm = confusion_matrix(
                labels.astype(np.int64),
                predictions.astype(np.int64),
                labels=classes)

            with open(confmat_path + '.txt', 'w') as fid:
                np.savetxt(fid, (classes + 1).reshape([1, -1]), fmt='%4d', delimiter=' ')
                np.savetxt(fid, cm, fmt='%4d', delimiter=' ')
                fid.write('\n')

            with open(confmat_path + '_per-class.txt', 'w') as fid:
                pc_acc = 100. * np.diag(cm) / np.sum(cm, axis=1)
                np.savetxt(fid, pc_acc, fmt='%5.1f', delimiter=',')

                invalid_class_inds = np.where(np.isnan(pc_acc))[0]
                valid_class_inds = np.delete(classes, invalid_class_inds)
                fid.write('average: {:5.1f}\n'.format(pc_acc[valid_class_inds].mean()))

            np.savez(confmat_path + '.npz', cm=cm, pc_acc=pc_acc)

        return acc, predictions
