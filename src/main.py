import os
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad

import modules
import utils
import classifiers
import data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['CUB', 'AWA1', 'SUN'])
parser.add_argument('--feature_norm', type=str, default='none', choices=['none', 'l2', 'unit-gaussian'])
parser.add_argument('--exp_dir', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--mode', type=str, default='test', choices=['validation', 'test'])
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
# generator arguments
parser.add_argument('--gen_type', type=str, default='latent_noise', choices=['latent_noise', 'attribute_concat'])
parser.add_argument('--d_noise', type=int)
parser.add_argument('--n_g_hlayer', type=int)
parser.add_argument('--n_g_hunit', type=int)
parser.add_argument('--normalize_noise', type=int, default=0, choices=[0, 1])
parser.add_argument('--dp_g', type=float, default=0.0)
parser.add_argument('--leakiness_g', type=float, default=0.2)
# discriminator arguments
parser.add_argument('--n_d_hlayer', type=int)
parser.add_argument('--n_d_hunit', type=int)
parser.add_argument('--d_normalize_ft', type=int, default=0, choices=[0, 1])
parser.add_argument('--dp_d', type=float, default=0.0)
parser.add_argument('--leakiness_d', type=float, default=0.2)
parser.add_argument('--n_d_iter', type=int)
parser.add_argument('--L', type=float)
# optimizer arguments
parser.add_argument('--gan_optim_lr_g', type=float)
parser.add_argument('--gan_optim_lr_d', type=float)
parser.add_argument('--gan_optim_beta1', type=float, default=0.0)
parser.add_argument('--gan_optim_beta2', type=float, default=0.9)
parser.add_argument('--gan_optim_wd', type=float, default=0.0)
# gmn arguments
parser.add_argument('--clf_type', type=str, default='bilinear-comp', choices=['bilinear-comp', 'multilayer-comp', 'mlp'])
parser.add_argument('--clf_optim_type', type=str, default='sgd', choices=['sgd', 'adam'])
parser.add_argument('--clf_lr', type=float, default=0.0)
parser.add_argument('--clf_wd', type=float, default=0.0)
parser.add_argument('--clf_mom', type=float, default=0.9)
parser.add_argument('--clf_reset_iter', type=int)
parser.add_argument('--n_gm_iter', type=int)
parser.add_argument('--per_class_batch_size', type=int)
parser.add_argument('--gm_fake_repeat', type=int)
parser.add_argument('--Q', type=float)
parser.add_argument('--Z', type=float)
# fcls arguments
parser.add_argument('--pretrained_clf_ckpt', type=str)
parser.add_argument('--C', type=float)
# cyclewgan arguments
parser.add_argument('--pretrained_regg_ckpt', type=str)
parser.add_argument('--R', type=float)
# misc
parser.add_argument('--batch_size', type=int)
parser.add_argument('--n_iter', type=int)
parser.add_argument('--n_ckpt', type=int)
parser.add_argument('--seed', type=int, default=67)
args = parser.parse_args()

np.set_printoptions(linewidth=180, precision=4, suppress=True)
torch.set_printoptions(linewidth=180, precision=4)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

FN = torch.from_numpy
dil = data.index_labels
join = os.path.join
logger = logging.getLogger()

def main():

    utils.prepare_directory(args.exp_dir, force_delete=False)
    utils.init_logger(join(args.exp_dir, 'program.log'))
    utils.write_args(args)

    # **************************************** load dataset ****************************************
    dset = data.XianDataset(args.data_dir, args.mode, feature_norm=args.feature_norm)
    _X_s_tr = FN(dset.X_s_tr).to(args.device)
    _Y_s_tr_ix = FN(dil(dset.Y_s_tr, dset.Cs)).to(args.device) # indexed labels
    _Ss = FN(dset.Sall[dset.Cs]).to(args.device)
    _Su = FN(dset.Sall[dset.Cu]).to(args.device)
    if args.d_noise == 0: args.d_noise = dset.d_attr

    # **************************************** create data loaders ****************************************
    _sampling_weights = None
    if args.dataset != 'SUN':
        _sampling_weights = data.compute_sampling_weights(dil(dset.Y_s_tr, dset.Cs)).to(args.device)
    xy_iter = data.Iterator(
        [_X_s_tr, _Y_s_tr_ix],
        args.batch_size,
        sampling_weights=_sampling_weights)
    label_iter = data.Iterator([torch.arange(dset.n_Cs, device=args.device)], args.batch_size)
    class_iter = data.Iterator([torch.arange(dset.n_Cs)], 1)

    # **************************************** per-class means and stds ****************************************
    # per class samplers and first 2 class moments
    per_class_iters = []
    Xs_tr_mean, Xs_tr_std = [], []
    Xs_te_mean, Xs_te_std = [], []
    Xu_te_mean, Xu_te_std = [], []
    for c_ix, c in enumerate(dset.Cs):
        # training samples of seen classes
        _inds = np.where(dset.Y_s_tr == c)[0]
        assert _inds.shape[0] > 0
        _X = dset.X_s_tr[_inds]
        Xs_tr_mean.append(_X.mean(axis=0, keepdims=True))
        Xs_tr_std.append(_X.std(axis=0, keepdims=True))

        if args.n_gm_iter > 0:
            _y = np.ones([_inds.shape[0]], np.int64) * c_ix
            per_class_iters.append(
                data.Iterator(
                    [FN(_X).to(args.device), FN(_y).to(args.device)],
                    args.per_class_batch_size ))

        # test samples of seen classes
        _inds = np.where(dset.Y_s_te == c)[0]
        assert _inds.shape[0] > 0
        _X = dset.X_s_te[_inds]
        Xs_te_mean.append(_X.mean(axis=0, keepdims=True))
        Xs_te_std.append(_X.std(axis=0, keepdims=True))

    # test samples of unseen classes
    for c_ix, c in enumerate(dset.Cu):
        _inds = np.where(dset.Y_u_te == c)[0]
        assert _inds.shape[0] > 0
        _X = dset.X_u_te[_inds]
        Xu_te_mean.append(_X.mean(axis=0, keepdims=True))
        Xu_te_std.append(_X.std(axis=0, keepdims=True))
    del _X, _inds, c_ix, c

    Xs_tr_mean = FN(np.concatenate(Xs_tr_mean, axis=0)).to(args.device)
    Xs_tr_std = FN(np.concatenate(Xs_tr_std, axis=0)).to(args.device)
    Xs_te_mean = FN(np.concatenate(Xs_te_mean, axis=0)).to(args.device)
    Xs_te_std = FN(np.concatenate(Xs_te_std, axis=0)).to(args.device)
    Xu_te_mean = FN(np.concatenate(Xu_te_mean, axis=0)).to(args.device)
    Xu_te_std = FN(np.concatenate(Xu_te_std, axis=0)).to(args.device)

    # **************************************** create networks ****************************************
    g_net = modules.get_generator(args.gen_type)(
        dset.d_attr, args.d_noise, args.n_g_hlayer, args.n_g_hunit, args.normalize_noise, args.dp_g, args.leakiness_g).to(args.device)
    g_optim = optim.Adam(
        g_net.parameters(), args.gan_optim_lr_g, betas=(args.gan_optim_beta1, args.gan_optim_beta2), weight_decay=args.gan_optim_wd)

    d_net = modules.ConditionalDiscriminator(
        dset.d_attr, args.n_d_hlayer, args.n_d_hunit, args.d_normalize_ft, args.dp_d, args.leakiness_d).to(args.device)
    d_optim = optim.Adam(
        d_net.parameters(), args.gan_optim_lr_d, betas=(args.gan_optim_beta1, args.gan_optim_beta2), weight_decay=args.gan_optim_wd)
    start_it = 1

    utils.model_info(g_net, 'g_net', args.exp_dir)
    utils.model_info(d_net, 'd_net', args.exp_dir)

    if args.n_gm_iter > 0:
        if args.clf_type == 'bilinear-comp':
            clf = classifiers.BilinearCompatibility(dset.d_ft, dset.d_attr, args)
        elif args.clf_type == 'mlp':
            clf = classifiers.MLP(dset.d_ft, dset.n_Cs, args)
        utils.model_info(clf.net, 'clf', args.exp_dir)

    pret_clf = None
    if os.path.isfile(args.pretrained_clf_ckpt):
        logger.info('Loading pre-trained {} checkpoint at {} ...'.format(args.clf_type, args.pretrained_clf_ckpt))
        ckpt = torch.load(args.pretrained_clf_ckpt, map_location=args.device)
        pret_clf = classifiers.BilinearCompatibility(dset.d_ft, dset.d_attr, args)
        pret_clf.net.load_state_dict(ckpt[args.clf_type])
        pret_clf.net.eval()
        for p in pret_clf.net.parameters(): p.requires_grad = False

    pret_regg = None
    if os.path.isfile(args.pretrained_regg_ckpt):
        logger.info('Loading pre-trained regressor checkpoint at {} ...'.format(args.pretrained_regg_ckpt))
        ckpt = torch.load(args.pretrained_regg_ckpt, map_location=args.device)
        pret_regg = classifiers.Regressor(args, dset.d_ft, dset.d_attr)
        pret_regg.net.load_state_dict(ckpt['regressor'])
        pret_regg.net.eval()
        for p in pret_regg.net.parameters(): p.requires_grad = False


    training_log_titles = [
        'd/loss',
        'd/real',
        'd/fake',
        'd/penalty',
        'gm/loss',
        'gm/real_loss',
        'gm/fake_loss',
        'g/fcls_loss',
        'g/cycle_loss',
        'clf/train_loss',
        'clf/train_acc',
        'mmad/X_s_tr',
        'mmad/X_s_te',
        'mmad/X_u_te',
        'smad/X_s_tr',
        'smad/X_s_te',
        'smad/X_u_te',
    ]
    if args.n_gm_iter > 0:
        training_log_titles.extend(
            ['grad-cossim/{}'.format(n) for n, p in clf.net.named_parameters()])
        training_log_titles.extend(
            ['grad-mse/{}'.format(n) for n, p in clf.net.named_parameters()])
    training_logger = utils.Logger(
        os.path.join(args.exp_dir, 'training-logs'),
        'logs',
        training_log_titles)

    t0 = time.time()

    logger.info('penguenler olmesin')
    for it in range(start_it, args.n_iter + 1):

        # **************************************** Discriminator updates ****************************************
        for p in d_net.parameters(): p.requires_grad = True
        for p in g_net.parameters(): p.requires_grad = False
        for _ in range(args.n_d_iter):
            x_real, y_ix = next(xy_iter)
            s = _Ss[y_ix]
            x_fake = g_net(s)

            d_real = d_net(x_real, s).mean()
            d_fake = d_net(x_fake, s).mean()
            d_penalty = modules.gradient_penalty(d_net, x_real, x_fake, s)
            d_loss = d_fake - d_real + args.L * d_penalty

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            training_logger.update_meters(
                ['d/real', 'd/fake', 'd/loss', 'd/penalty'],
                [d_real.mean().item(), d_fake.mean().item(), d_loss.item(), d_penalty.item()],
                x_real.size(0))

        # **************************************** Generator updates ****************************************
        for p in d_net.parameters(): p.requires_grad = False
        for p in g_net.parameters(): p.requires_grad = True
        g_optim.zero_grad()

        [y_fake] = next(label_iter)
        s = _Ss[y_fake]
        x_fake = g_net(s)

        # wgan loss
        d_fake = d_net(x_fake, s).mean()
        g_wganloss = - d_fake

        # f-cls loss
        fcls_loss = 0.0
        if pret_clf is not None:
            fcls_loss = pret_clf.loss(x_fake, _Ss, y_fake)
            training_logger.update_meters(
                ['g/fcls_loss'],
                [fcls_loss.item()],
                x_fake.size(0))

        # cycle-loss
        cycle_loss = 0.0
        if pret_regg is not None:
            cycle_loss = pret_regg.loss(x_fake, s)
            training_logger.update_meters(
                ['g/cycle_loss'],
                [cycle_loss.item()],
                x_fake.size(0))

        g_loss = args.C * fcls_loss + args.R * cycle_loss + g_wganloss
        g_loss.backward()

        # gmn iterations
        for _ in range(args.n_gm_iter):
            c = next(class_iter)[0].item()
            x_real, y_real = next(per_class_iters[c])
            y_fake = y_real.detach().repeat(args.gm_fake_repeat)
            s = _Ss[y_fake]
            x_fake = g_net(s)

            # gm loss
            clf.net.zero_grad()
            if args.clf_type == 'bilinear-comp':
                real_loss = clf.loss(x_real, _Ss, y_real)
                fake_loss = clf.loss(x_fake, _Ss, y_fake)
            elif args.clf_type == 'mlp': 
                real_loss = clf.loss(x_real, y_real)
                fake_loss = clf.loss(x_fake, y_fake)

            grad_cossim = []
            grad_mse = []
            for n, p in clf.net.named_parameters():
                # if len(p.shape) == 1: continue

                real_grad = grad([real_loss],
                                 [p],
                                 create_graph=True,
                                 only_inputs=True)[0]
                fake_grad = grad([fake_loss],
                                 [p],
                                 create_graph=True,
                                 only_inputs=True)[0]

                if len(p.shape) > 1:
                    _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
                else:
                    _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)

                # _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
                _mse = F.mse_loss(fake_grad, real_grad)
                grad_cossim.append(_cossim)
                grad_mse.append(_mse)

                training_logger.update_meters(
                    ['grad-cossim/{}'.format(n), 'grad-mse/{}'.format(n)],
                    [_cossim.item(), _mse.item()],
                    x_real.size(0))

            grad_cossim = torch.stack(grad_cossim)
            grad_mse = torch.stack(grad_mse)
            gm_loss = (1.0 - grad_cossim).sum() * args.Q + grad_mse.sum() * args.Z
            gm_loss.backward()

            training_logger.update_meters(
                ['gm/real_loss', 'gm/fake_loss'],
                [real_loss.item(), fake_loss.item()],
                x_real.size(0))

        g_optim.step()

        # **************************************** Classifier update ****************************************
        if args.n_gm_iter > 0:
            if it % args.clf_reset_iter == 0:
                if args.clf_reset_iter == 1:
                    # no need to generate optimizer each time
                    clf.init_params()
                else:
                    clf.reset()
            else:
                x, y_ix = next(xy_iter)
                if args.clf_type == 'bilinear-comp':
                    clf_acc, clf_loss = clf.train_step(x, _Ss, y_ix)
                else:
                    clf_acc, clf_loss = clf.train_step(x, y_ix)
                training_logger.update_meters(
                    ['clf/train_loss', 'clf/train_acc'],
                    [clf_loss, clf_acc],
                    x.size(0))

        # **************************************** Log ****************************************
        if it % 1000 == 0:
            g_net.eval()

            # synthesize samples for seen classes and compute their first 2 moments
            Xs_fake_mean, Xs_fake_std = [], []
            with torch.no_grad():
                for c in range(dset.n_Cs):
                    y = torch.ones(256, device=args.device, dtype=torch.long) * c
                    a = _Ss[y]
                    x_fake = g_net(a)
                    Xs_fake_mean.append(x_fake.mean(dim=0, keepdim=True))
                    Xs_fake_std.append(x_fake.std(dim=0, keepdim=True))
            Xs_fake_mean = torch.cat(Xs_fake_mean)
            Xs_fake_std = torch.cat(Xs_fake_std)

            # synthesize samples for unseen classes and compute their first 2 moments
            def compute_firsttwo_moments(S, C):
                X_mean, X_std = [], []
                with torch.no_grad():
                    for c in range(dset.n_Cu):
                        y = torch.ones(256, device=args.device, dtype=torch.long) * c
                        a = _Su[y]
                        x_fake = g_net(a)
                        X_mean.append(x_fake.mean(dim=0, keepdim=True))
                        X_std.append(x_fake.std(dim=0, keepdim=True))
                X_mean = torch.cat(X_mean)
                X_std = torch.cat(X_std)

            Xu_fake_mean, Xu_fake_std = [], []
            with torch.no_grad():
                for c in range(dset.n_Cu):
                    y = torch.ones(256, device=args.device, dtype=torch.long) * c
                    a = _Su[y]
                    x_fake = g_net(a)
                    Xu_fake_mean.append(x_fake.mean(dim=0, keepdim=True))
                    Xu_fake_std.append(x_fake.std(dim=0, keepdim=True))
            Xu_fake_mean = torch.cat(Xu_fake_mean)
            Xu_fake_std = torch.cat(Xu_fake_std)

            g_net.train()

            training_logger.update_meters(
                ['mmad/X_s_tr', 'smad/X_s_tr',
                 'mmad/X_s_te', 'smad/X_s_te',
                 'mmad/X_u_te', 'smad/X_u_te' ],
                [torch.abs(Xs_tr_mean - Xs_fake_mean).sum(dim=1).mean().item(),
                 torch.abs(Xs_tr_std - Xs_fake_std).sum(dim=1).mean().item(),
                 torch.abs(Xs_te_mean - Xs_fake_mean).sum(dim=1).mean().item(),
                 torch.abs(Xs_te_std - Xs_fake_std).sum(dim=1).mean().item(),
                 torch.abs(Xu_te_mean - Xu_fake_mean).sum(dim=1).mean().item(),
                 torch.abs(Xu_te_std - Xu_fake_std).sum(dim=1).mean().item() ])

            training_logger.flush_meters(it)

            elapsed = time.time() - t0
            per_iter = elapsed / it
            apprx_rem = (args.n_iter - it) * per_iter
            logging.info('Iter:{:06d}/{:06d}, '\
                         '[ET:{:.1e}(min)], ' \
                         '[IT:{:.1f}(ms)], ' \
                         '[REM:{:.1e}(min)]'.format(
                            it, args.n_iter, elapsed / 60., per_iter * 1000., apprx_rem / 60))

        if it % 10000 == 0:
            utils.save_checkpoint(
                {
                    'g_net': g_net.state_dict(),
                    'd_net': d_net.state_dict(),
                    'g_optim': g_optim.state_dict(),
                    'd_optim': d_optim.state_dict(),
                    'iteration': it
                },
                args.exp_dir,
                None,
                it if it % (args.n_iter // args.n_ckpt) == 0 else None,
            )
    
    training_logger.close()

if __name__ == "__main__":
    main()
