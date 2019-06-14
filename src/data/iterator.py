import numpy as np
import math
import torch
import logging
logger = logging.getLogger()

class Iterator():
    """
    Iterator for list of tensors whose first dimension match.
    """

    def __init__(self, tensors, batch_size, 
                       allow_smaller=True, 
                       shuffle=True,
                       sampling_weights=None,
                       continuous=True):
        self.tensors = tensors
        self.batch_size = batch_size
        self.allow_smaller = allow_smaller
        self.shuffle = shuffle and sampling_weights is None 
        self.sampling_weights = sampling_weights
        self.continuous = continuous
        self.device = tensors[0].device

        # number of elements in each tensor should be equal and a positive number
        n_elems = [len(t) for t in tensors]
        assert np.all(np.equal(n_elems, n_elems[0]))
        self.n_sample = n_elems[0]
        assert self.n_sample > 0

        while self.n_sample < self.batch_size:
            self.tensors = [t.repeat(2, *([1] * (len(t.shape) - 1))) for t in self.tensors]
            self.n_sample *= 2
            logger.info('Tensors are repeated, new sizes:{}'.format([t.shape for t in self.tensors]))

        self._s_ix = 0 # index of sample that will be fetched as the first sample in next_batch
        self._order = torch.zeros(self.n_sample, dtype=torch.long, device=self.device) # order of samples fetched in an epoch
        self.reset_batch_order()

    def __len__(self):
        return math.ceil(self.n_sample / self.batch_size)

    def __iter__(self):
        return self

    def _check_new_epoch(self):
        if self.allow_smaller:
            # check whether there is no not-fetched sample left
            return self._s_ix >= self.n_sample
        else:
            # check whether number of remaining not-fetched samples less than the batch size
            return self.n_sample - self._s_ix < self.batch_size

    def reset_batch_order(self):
        self._s_ix = 0
        if self.sampling_weights is not None:
            torch.multinomial(self.sampling_weights, self.n_sample, replacement=True, out=self._order)
        elif self.shuffle:
            torch.randperm(self.n_sample, out=self._order)
        else:
            torch.arange(self.n_sample, out=self._order)
        
    def __next__(self):
        new_epoch = self._check_new_epoch()
        if new_epoch:
            self.reset_batch_order()
            if not self.continuous:
                raise StopIteration

        inds = self._order[self._s_ix : self._s_ix + self.batch_size]
        self._s_ix += self.batch_size
        batch = [t[inds] for t in self.tensors]
        return batch
        

def compute_sampling_weights(labels):
    classes = np.unique(labels)
    assert classes.max() == (classes.shape[0] - 1)
    n_samples_per_class = np.array([len(np.where(labels == c)[0]) for c in classes])
    class_weights = 1. / n_samples_per_class
    sample_weights = np.array([class_weights[l] for l in labels])
    sample_weights = torch.from_numpy(sample_weights).float()
    return sample_weights
