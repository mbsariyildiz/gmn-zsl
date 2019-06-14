import os
import logging
import numpy as np
from scipy.io import loadmat

join = os.path.join
logger = logging.getLogger()


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class XianDataset(dotdict):

    def __init__(self, data_dir, mode, feature_norm='none', verbose=True):
        super().__init__()
        assert feature_norm == 'none' or \
               feature_norm == 'l2' or \
               feature_norm == 'unit-gaussian'

        feature_label_file = join(data_dir, 'res101.mat')
        embedding_file = join(data_dir, 'att_splits.mat')
        split_file = join(data_dir, 'att_splits.mat')

        self.Sall = loadmat(embedding_file)['att'].T.astype('float32')
        self.Sall /= np.linalg.norm(self.Sall, axis=1, keepdims=True)
        if verbose:
            logger.info ('Sall average norm: {}'.format(
                np.linalg.norm(self.Sall, axis=1).mean()))            

        data = loadmat(feature_label_file)
        self.Xall = data['features'].astype('float32')
        if self.Xall.shape[0] < self.Xall.shape[1]: self.Xall = self.Xall.T # if transposed
        if feature_norm == 'l2':
            self.Xall /= np.linalg.norm(self.Xall, axis=1, keepdims=True)
            
        self.Yall = np.int64(data['labels']).ravel()
        self.Yall = self.Yall - self.Yall.min() # labels start from 0

        # training-validation-test splits
        locs = loadmat(split_file)
        for k,v in locs.items():
            if 'loc' in k:
                locs[k] = v.ravel() - 1

        if mode == 'validation':
            X_s_trte = self.Xall[locs['train_loc']]
            Y_s_trte = self.Yall[locs['train_loc']].ravel()

            # split the current training set into train + val
            # though it would probably be better to make uniform splits:
            #   - same number of samples for each class in each split
            n_trtel_sample = X_s_trte.shape[0]
            n_tr_sample = int(0.8 * n_trtel_sample)
            np.random.seed(42)
            order = np.random.permutation(n_trtel_sample)
            self.X_s_tr = X_s_trte[order[:n_tr_sample]]
            self.Y_s_tr = Y_s_trte[order[:n_tr_sample]]
            self.X_s_te = X_s_trte[order[n_tr_sample:]]
            self.Y_s_te = Y_s_trte[order[n_tr_sample:]]

            self.X_u_te = self.Xall[locs['val_loc']]
            self.Y_u_te = self.Yall[locs['val_loc']].ravel()

        elif mode == 'test':
            self.X_s_tr = self.Xall[locs['trainval_loc']]
            self.Y_s_tr = self.Yall[locs['trainval_loc']].ravel()

            self.X_s_te = self.Xall[locs['test_seen_loc']]
            self.Y_s_te = self.Yall[locs['test_seen_loc']].ravel()
            
            self.X_u_te = self.Xall[locs['test_unseen_loc']]
            self.Y_u_te = self.Yall[locs['test_unseen_loc']].ravel()

        if feature_norm == 'unit-gaussian':
            _mean = self.X_s_tr.mean(axis=0, keepdims=True)
            _std = self.X_s_tr.std(axis=0, keepdims=True).clip(min=1e-4)
            self.X_s_tr = (self.X_s_tr - _mean) / _std
            self.X_s_te = (self.X_s_te - _mean) / _std
            self.X_u_te = (self.X_u_te - _mean) / _std

        if verbose:
            logger.info ('X_s_tr average norm: {:.3f}'.format(np.linalg.norm(self.X_s_tr, axis=1).mean()))
            logger.info ('X_s_te average norm: {:.3f}'.format(np.linalg.norm(self.X_s_te, axis=1).mean()))
            logger.info ('X_u_te average norm: {:.3f}'.format(np.linalg.norm(self.X_u_te, axis=1).mean()))

        self.d_ft = self.Xall.shape[1]
        self.d_attr = self.Sall.shape[1]
        self.Call = np.unique(self.Yall)
        self.Cs = np.unique(np.concatenate([self.Y_s_tr, self.Y_s_te]))
        self.Cu = np.unique(self.Y_u_te)
        self.n_Call = self.Call.shape[0]
        self.n_Cs = self.Cs.shape[0]
        self.n_Cu = self.Cu.shape[0]

        if data_dir.endswith('CUB'):
            readfeatures_file = join(data_dir, 'cub_attributes_reed.npy')
            Rall = np.load(readfeatures_file).astype(np.float32)
            Rall = Rall / np.linalg.norm(Rall)
            self.Sall = np.concatenate([self.Sall, Rall], axis=1)
            self.d_attr = self.Sall.shape[1]
        
        if verbose:
            self.print_info()

    def print_info(self):
        logger.info ('%10s -- %20s -- %10s' % ('keys', 'shapes', 'types'))
        for k, v in self.items():
            if isinstance(v, np.ndarray):
                logger.info ('%10s -- %20s -- %10s' % (k, v.shape, v.dtype))


def index_labels(labels, classes, check=True):
    """
    Indexes labels in classes.

    Arg:
        labels:  [batch_size]
        classes: [n_class]
    """
    indexed_labels = np.searchsorted(classes, labels)
    if check:
        assert np.all(np.equal(classes[indexed_labels], labels))

    return indexed_labels
