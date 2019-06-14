import os
import argparse
import random
import logging
import numpy as np
import torch as th

import utils
import data
import classifiers

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--feature_norm', type=str, default='none', choices=['none', 'l2', 'unit-gaussian'])
parser.add_argument('--data_dir', type=str)
parser.add_argument('--exp_root', type=str)
parser.add_argument('--mode', type=str, default='test', choices=['test', 'validation'])
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
# classifier arguments
parser.add_argument('--lr', type=float, default=0.)
parser.add_argument('--wd', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=0)
parser.add_argument('--n_epoch', type=int, default=0)
args = parser.parse_args()

np.set_printoptions(linewidth=150, precision=4, suppress=True)
th.set_printoptions(linewidth=150, precision=4)

FN = th.from_numpy
join = os.path.join
logger = logging.getLogger()

utils.prepare_directory(args.exp_root, force_delete=True)
utils.init_logger(join(args.exp_root, 'program.log'))
utils.write_args(args)

dset = data.XianDataset(args.data_dir, args.mode, feature_norm=args.feature_norm)
_X_s_tr = FN(dset.X_s_tr).to(args.device)
_Y_s_tr = FN(dset.Y_s_tr).to(args.device)
_X_s_te = FN(dset.X_s_te).to(args.device)
_Y_s_te = FN(dset.Y_s_te).to(args.device)
_X_u_te = FN(dset.X_u_te).to(args.device)
_Y_u_te = FN(dset.Y_u_te).to(args.device)
_Cu = FN(dset.Cu).to(args.device)
_Sall = FN(dset.Sall).to(args.device)

train_iter = data.Iterator(
    [_X_s_tr, _Y_s_tr],
    args.batch_size,
    shuffle=True,
    continuous=False)

seen_test_iter = data.Iterator(
    [_X_s_te, _Y_s_te],
    1000,
    shuffle=False,
    continuous=False)

unseen_test_iter = data.Iterator(
    [_X_u_te, _Y_u_te],
    1000,
    shuffle=False,
    continuous=False)


seeds = [123, 67, 1234, 96, 4444]
n_trial = len(seeds)
train_logs = np.zeros([n_trial, args.n_epoch, 2], 'float32')
accs = np.zeros([n_trial, args.n_epoch, 4], 'float32')

for trial, seed in enumerate(seeds):
    logger.info ('trial {} / {} ... '.format(trial + 1, n_trial))
    
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    exp_dir = join(args.exp_root, 'regg-model_{:03d}'.format(trial))
    utils.prepare_directory(exp_dir)
    cm_zsl_path = join(exp_dir, 'cm_zsl')
    cm_gzslu_path = join(exp_dir, 'cm_gzslu')
    cm_gzsls_path = join(exp_dir, 'cm_gzsls')

    logger.info ('Initializing a regressor model ...')
    regg = classifiers.Regressor(args, dset.d_ft, dset.d_attr) 
    utils.model_info(regg.net, 'regg', exp_dir)
    for epoch in range(args.n_epoch):

        train_loss, train_acc = regg.train_epoch(train_iter, _Sall)
        train_logs[trial, epoch, :] = train_loss, train_acc

        acc_zsl, _ = regg.test(unseen_test_iter, _Sall, _Cu, cm_zsl_path)
        acc_gzslu, _ = regg.test(unseen_test_iter, _Sall, confmat_path=cm_gzslu_path)
        acc_gzsls, _ = regg.test(seen_test_iter, _Sall, confmat_path=cm_gzsls_path)
        acc_gzslh = 2. * acc_gzslu * acc_gzsls / (acc_gzslu + acc_gzsls)
        accs[trial, epoch, :] = acc_zsl, acc_gzslu, acc_gzsls, acc_gzslh
    
    utils.save_checkpoint(
        {'regressor': regg.net.state_dict()},
         exp_dir)

    del regg
    th.cuda.empty_cache()

train_loss_mean  = train_logs[:, :, 0].mean(axis=0)
train_loss_std   = train_logs[:, :, 0].std(axis=0)
train_acc_mean  = train_logs[:, :, 1].mean(axis=0)
train_acc_std   = train_logs[:, :, 1].std(axis=0)
zsl_mean   = accs[:, :, 0].mean(axis=0)
zsl_std    = accs[:, :, 0].std(axis=0)
gzslu_mean = accs[:, :, 1].mean(axis=0)
gzslu_std  = accs[:, :, 1].std(axis=0)
gzsls_mean = accs[:, :, 2].mean(axis=0)
gzsls_std  = accs[:, :, 2].std(axis=0)
gzslh_mean = accs[:, :, 3].mean(axis=0)
gzslh_std  = accs[:, :, 3].std(axis=0)

log_titles = [
    'train-l2/mean',
    'train-l2/std',
    'train-acc/mean',
    'train-acc/std',
    'zsl/mean',
    'zsl/std',
    'gzslu/mean',
    'gzslu/std',
    'gzsls/mean',
    'gzsls/std',
    'gzslh/mean',
    'gzslh/std'
]
result_logger = utils.Logger(
    join(args.exp_root, 'avg-scores'),
    'logs',
    log_titles)

for epoch in range(args.n_epoch):
    log_values = [
        train_loss_mean[epoch].item(),
        train_loss_std[epoch].item(),
        train_acc_mean[epoch].item(),
        train_acc_std[epoch].item(),
        zsl_mean[epoch].item(),
        zsl_std[epoch].item(),
        gzslu_mean[epoch].item(),
        gzslu_std[epoch].item(),
        gzsls_mean[epoch].item(),
        gzsls_std[epoch].item(),
        gzslh_mean[epoch].item(),
        gzslh_std[epoch].item()
    ]

    result_logger.append(log_titles, log_values, epoch)

result_logger.close()

