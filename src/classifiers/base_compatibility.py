import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import utils
import data

logger = logging.getLogger()


class _BaseCompatibility():

    def __init__(self, d_ft, d_emb, args):
        self.d_ft = d_ft
        self.d_emb = d_emb
        self.n_hlayer = args.clf_n_hlayer if 'clf_n_hlayer' in args else 0
        self.n_hunit = args.clf_n_hunit if 'clf_n_hunit' in args else 0
        self.dp = args.clf_dp if 'clf_dp' in args else 0.
        self.opt_type = args.clf_optim_type if 'clf_optim_type' in args else None
        self.learning_rate = args.clf_lr if 'clf_lr' in args else 0
        self.learning_rate_decay = args.clf_lr_decay if 'clf_lr_decay' in args else 1.0
        self.momentum = args.clf_mom if 'clf_mom' in args else 0.0
        self.weight_decay = args.clf_wd if 'clf_wd' in args else 0
        self.device = args.device if 'device' in args else 'cpu'

        # this should be initialized by a child class
        self.net = None
        self.criterion = nn.CrossEntropyLoss()

    def reset(self):
        self.init_params()

        if self.opt_type == 'sgd':
            self.optim = optim.SGD(
                self.net.parameters(),
                self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay)

        elif self.opt_type == 'adam':
            self.optim = optim.Adam(
                self.net.parameters(),
                self.learning_rate,
                weight_decay=self.weight_decay)
        
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optim, 1, gamma=self.learning_rate_decay)

    def init_params(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.normal_(m.bias, 0.0, 0.0)

    def loss(self, x, S, y):
        logits = self.net(x, S)
        loss = self.criterion(logits, y)
        return loss

    def train_step(self, x, S, y):
        logits = self.net(x, S)
        loss = self.criterion(logits, y)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        prec1 = utils.accuracy(logits.detach(), y.detach())[0]
        return prec1.item(), loss.item()

    def train_epoch(self, iterator, S):
        train_loss = utils.AverageMeter()
        train_acc = utils.AverageMeter()

        self.net.train()

        for x, y in iterator:
            prec1, loss = self.train_step(x, S, y)
            train_acc.update(prec1, x.size(0))
            train_loss.update(loss, x.size(0))

        self.lr_scheduler.step()

        return train_loss.avg, train_acc.avg

    def test(self, iterator, S, C=None, confmat_path=''):
        labels = torch.zeros(iterator.n_sample, dtype=torch.int64, device=self.device)
        predictions = torch.zeros(iterator.n_sample, dtype=torch.int64, device=self.device)

        s_ix = 0
        self.net.eval()

        with torch.no_grad():

            for x, y in iterator:
                bs = x.size(0)

                logits = self.net(x, S)
                if C is not None:
                    # take the logits of the classes in C (for ZSL evaluation,)
                    logits = logits[:, C]
                _, preds = logits.max(dim=1)
                labels[s_ix : s_ix + bs] = y
                predictions[s_ix:s_ix+bs] = preds
                s_ix += bs

        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        if C is not None:
            # map labels to be in [0, C.shape[0]]
            labels = data.index_labels(labels, C.cpu().numpy())

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
                np.savetxt(fid, classes+1, fmt='%4d', delimiter=' ')
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

