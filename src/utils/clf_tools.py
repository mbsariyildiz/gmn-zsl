import numpy as np

def normalized_acc(trues, preds):
    """
    Given true and predicted labels, computes average class-based accuracy.
    """

    # class labels in ground-truth samples
    classes = np.unique(trues)
    # class-based accuracies
    cb_accs = np.zeros(classes.shape, np.float32)

    for i, label in enumerate(classes):
        inds_ci = np.where(trues == label)[0]

        cb_accs[i] = np.mean(
            np.equal(
                trues[inds_ci],
                preds[inds_ci]
            ).astype(np.float32)
        )

    return np.mean(cb_accs) * 100.

def accuracy(logits, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res
    