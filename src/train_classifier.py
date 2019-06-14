import os
import argparse
import random
import logging
import numpy as np
import torch as th

import utils
import data
import classifiers
import modules

np.set_printoptions(linewidth=150, precision=4, suppress=True)
th.set_printoptions(linewidth=150, precision=4)
FN = th.from_numpy
join = os.path.join
logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--feature_norm', type=str, default='none', choices=['none', 'l2', 'unit-gaussian'])
parser.add_argument('--data_dir', type=str)
parser.add_argument('--exp_root', type=str)
parser.add_argument('--mode', type=str, default='test', choices=['test', 'validation'])
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
# classifier arguments
parser.add_argument('--clf_type', type=str, choices=['bilinear-comp', 'multilayer-comp', 'mlp'])
parser.add_argument('--clf_n_hlayer', type=int, default=0)
parser.add_argument('--clf_n_hunit', type=int, default=0)
parser.add_argument('--clf_optim_type', type=str, default='sgd', choices=['sgd', 'adam'])
parser.add_argument('--clf_lr', type=float, default=0.)
parser.add_argument('--clf_lr_decay', type=float, default=1.)
parser.add_argument('--clf_wd', type=float, default=0.)
parser.add_argument('--clf_mom', type=float, default=0.9)
parser.add_argument('--uniform_sampling', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=0)
parser.add_argument('--n_epoch', type=int, default=0)
# generator arguments
parser.add_argument('--generator_ckpt', type=str, default='')
parser.add_argument('--gen_type', type=str, default='', choices=['', 'latent_noise', 'attribute_concat'])
parser.add_argument('--d_noise', type=int)
parser.add_argument('--n_g_hlayer', type=int)
parser.add_argument('--n_g_hunit', type=int)
parser.add_argument('--leakiness_g', type=float, default=0.2)
parser.add_argument('--dp_g', type=float, default=0.0)
parser.add_argument('--n_synth_U', type=int)
parser.add_argument('--n_synth_S', type=int, default=0)
parser.add_argument('--normalize_noise', type=int, default=0, choices=[0, 1])
args = parser.parse_args()

utils.prepare_directory(args.exp_root, force_delete=True)
utils.init_logger(join(args.exp_root, 'program.log'))

if not th.cuda.is_available():
    args.device = 'cpu'
    import psutil
    n_cpu = psutil.cpu_count()
    n_cpu_to_use = n_cpu // 4
    logger.info('{} CPUs found in system and {} of those will be used.'.format(n_cpu, n_cpu_to_use))
    th.set_num_threads(n_cpu_to_use)
    os.environ['MKL_NUM_THREADS'] = str(n_cpu_to_use)
    os.environ['KMP_AFFINITY'] = 'compact'

utils.write_args(args)

dset = data.XianDataset(args.data_dir, args.mode, feature_norm=args.feature_norm)
_X_s_tr = FN(dset.X_s_tr).to(args.device)
_Y_s_tr = FN(dset.Y_s_tr).to(args.device)
_Y_s_tr_ix = FN(data.index_labels(dset.Y_s_tr, dset.Cs)).to(args.device)
_X_s_te = FN(dset.X_s_te).to(args.device)
_Y_s_te = FN(dset.Y_s_te).to(args.device)
_Cs = FN(dset.Cs).to(args.device)
_X_u_te = FN(dset.X_u_te).to(args.device)
_Y_u_te = FN(dset.Y_u_te).to(args.device)
_Cu = FN(dset.Cu).to(args.device)
_Sall = FN(dset.Sall).to(args.device)
_Ss = _Sall[_Cs]

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

def synthesize_samples(n_synth):
    """
    Draw samples from the pretrained generator model whose ckpt is
    stored at args.generator_ckpt. Then returns the synthetic dataset.

    args:
        n_synth : number of samples to synthesize for each class
    """
    assert _Sall.shape[0] == n_synth.shape[0]

    g_net = modules.get_generator(args.gen_type)(
        dset.d_attr, args.d_noise, args.n_g_hlayer, args.n_g_hunit, args.normalize_noise, args.dp_g, args.leakiness_g).to(args.device)
    logger.info('Loading generator checkpoint at {}'.format(args.generator_ckpt))
    ckpt = th.load(args.generator_ckpt, map_location=args.device)
    g_net.load_state_dict(ckpt['g_net'])
    g_net.eval()

    samples = []
    labels = []
    N = n_synth.sum().item() # total number of samples to generate

    # TODO: initialize samples and labels here

    with th.no_grad():
        for c, n in enumerate(n_synth): # c: class label
                                        # n: number of samples to synthesize for this class
            six = 0

            while six < n:
                n_req = n - six
                bs = min(args.batch_size, n_req)
                y = th.ones(bs, device=args.device, dtype=th.long) * c
                a = _Sall[y]
                samples.append(g_net(a))
                labels.append(y)
                six += bs

            assert six == n

    samples = th.cat(samples)
    labels = th.cat(labels)
    assert samples.shape[0] == N
    assert labels.shape[0] == N
    return samples, labels

if args.d_noise == 0: args.d_noise = dset.d_attr

seeds = [123, 67, 1234, 96, 4444] # 
n_trials = len(seeds)
accs = np.zeros([n_trials, args.n_epoch, 4], 'float32')

for trial, seed in enumerate(seeds):
    logger.info ('trial {} / {} ... '.format(trial + 1, n_trials))

    exp_dir = join(args.exp_root, 'clf-model_{:03d}'.format(trial))
    utils.prepare_directory(exp_dir)

    cm_zsl_path = join(exp_dir, 'cm_zsl')
    cm_gzslu_path = join(exp_dir, 'cm_gzslu')
    cm_gzsls_path = join(exp_dir, 'cm_gzsls')

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    Xtr = _X_s_tr.clone()
    Ytr = _Y_s_tr.clone() if args.clf_type == 'mlp' else _Y_s_tr_ix.clone()
    Str = _Ss.clone()
    sampling_weights = None
    unseen_generated = False
    if os.path.isfile(args.generator_ckpt):
        assert args.n_synth_U > 0

        # number of training samples for each seen class
        nspc_str = np.array([len(np.where(dset.Y_s_tr == L)[0]) for L in np.unique(dset.Y_s_tr)])
        # array that denotes how many samples will be generated for each class
        n_synth = np.zeros([dset.n_Call], dtype=np.int64)
        n_synth[dset.Cs] = (args.n_synth_S - nspc_str).clip(min=0)
        n_synth[dset.Cu] = args.n_synth_U

        X_fake, Y_fake = synthesize_samples(n_synth)
        if args.feature_norm == 'l2':
            X_fake = th.nn.functional(X_fake, dim=1, p=2)
        logger.info ('{} samples generated'.format(X_fake.shape[0]))
        # verify that indeed n_synth samples are generated for the classes
        _ytmp = Y_fake.cpu().numpy()
        for _c in np.unique(_ytmp):
            _n = np.where(_ytmp == _c)[0].shape[0]
            assert _n == n_synth[_c]
        np.savetxt(
            join(exp_dir, 'n_synth'), n_synth, fmt='%d', delimiter=',')
        del _ytmp, _c, _n, n_synth

        Xtr = th.cat([Xtr, X_fake])
        # now we use un-indexed labels
        del Ytr
        Ytr = th.cat([_Y_s_tr, Y_fake])
        # and also all embeddings, not just the embeddings of seen classes
        del Str
        Str = _Sall.clone()
        unseen_generated = True
        logger.info ('Size of the pooled dataset:{}'.format(Xtr.shape[0]))
        del X_fake, Y_fake
        th.cuda.empty_cache()
        if args.uniform_sampling:
            sampling_weights = utils.compute_sampling_weights(
                Ytr.cpu().numpy()).to(args.device)


    train_iter = data.Iterator(
        [Xtr, Ytr],
        args.batch_size,
        shuffle=True,
        sampling_weights=sampling_weights,
        continuous=False)

    logger.info ('Initializing {} model ...'.format(args.clf_type))
    clf = None
    if args.clf_type == 'bilinear-comp':
        clf = classifiers.BilinearCompatibility(dset.d_ft, dset.d_attr, args)
    elif args.clf_type == 'mlp':
        clf = classifiers.MLP(dset.d_ft, dset.n_Call, args)
    elif args.clf_type == 'multilayer-comp':
        clf = classifiers.MultiLayerCompatibility(dset.d_ft, dset.d_attr, args)
    utils.model_info(clf.net, 'clf', exp_dir)
    for epoch in range(args.n_epoch):

        if args.clf_type == 'bilinear-comp' or args.clf_type == 'multilayer-comp':
            clf.train_epoch(train_iter, Str)
            acc_zsl, _ = clf.test(unseen_test_iter, _Sall, _Cu, cm_zsl_path)
            acc_gzslu, _ = clf.test(unseen_test_iter, _Sall, confmat_path=cm_gzslu_path)
            acc_gzsls, _ = clf.test(seen_test_iter, _Sall, confmat_path=cm_gzsls_path)
            acc_gzslh = 2. * acc_gzslu * acc_gzsls / (acc_gzslu + acc_gzsls)
            accs[trial, epoch, :] = acc_zsl, acc_gzslu, acc_gzsls, acc_gzslh
        else:
            clf.train_epoch(train_iter)
            acc_zsl, _ = clf.test(unseen_test_iter, _Cu, cm_zsl_path)
            acc_gzslu, _ = clf.test(unseen_test_iter, confmat_path=cm_gzslu_path)
            acc_gzsls, _ = clf.test(seen_test_iter, confmat_path=cm_gzsls_path)
            acc_gzslh = 2. * acc_gzslu * acc_gzsls / (acc_gzslu + acc_gzsls)
            accs[trial, epoch, :] = acc_zsl, acc_gzslu, acc_gzsls, acc_gzslh

    # utils.save_checkpoint(
    #     {'{}'.format(args.clf_type): clf.net.state_dict(),
    #      'Xtr': Xtr.cpu().numpy(),
    #      'Ytr': Ytr.cpu().numpy() },
    #      exp_dir)
    utils.save_checkpoint(
        {'{}'.format(args.clf_type): clf.net.state_dict()},
         exp_dir)

    del Xtr, Ytr, Str, train_iter, clf
    th.cuda.empty_cache()

zsl_mean   = accs[:, :, 0].mean(axis=0)
zsl_std    = accs[:, :, 0].std(axis=0)
gzslu_mean = accs[:, :, 1].mean(axis=0)
gzslu_std  = accs[:, :, 1].std(axis=0)
gzsls_mean = accs[:, :, 2].mean(axis=0)
gzsls_std  = accs[:, :, 2].std(axis=0)
gzslh_mean = accs[:, :, 3].mean(axis=0)
gzslh_std  = accs[:, :, 3].std(axis=0)

log_titles = [
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

