import os
import shutil
import logging
import torch

__all__ = [
    'init_logger',
    'save_checkpoint',
    'model_info',
    'write_args',
    'prepare_directory',
    'AverageMeter'
]

logger = logging.getLogger()

def init_logger(log_file):
    logger.setLevel(logging.INFO)

    log_formatter = logging.Formatter(
        fmt='%(asctime)s [%(threadName)s] [%(module)s] [%(funcName)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def save_checkpoint(state, dir, is_best=False, cur_iter=None):
    ckpt_file = os.path.join(dir, 'model.ckpt')
    torch.save(state, ckpt_file)
    if is_best:
        shutil.copyfile(ckpt_file, os.path.join(dir, 'model_best.ckpt'))
    if cur_iter is not None:
        shutil.copyfile(ckpt_file, os.path.join(dir, 'model_{:06d}.ckpt'.format(cur_iter)))

def model_info(model, model_name, save_dir=''):
    logger.info('Number of parameters in {}: {}'.format(
        model_name,
        sum([p.data.nelement() for p in model.parameters()]))) 

    def save_model_desc(model, path):
        with open(path, 'w') as fid:
            fid.write(str(model))

    if save_dir:
        save_model_desc(
            model, os.path.join(save_dir, '{}_desc.txt'.format(model_name)))

def write_args(FLAGS):
    # save all setup into a log file
    _dict = vars(FLAGS)
    _list = sorted(_dict.keys())

    logger.info('Arguments:')
    for _k in _list:
        logger.info('\t%s: %s' % (_k, _dict[_k]))

def prepare_directory(directory, force_delete=False):
    if os.path.exists(directory) and (not force_delete):
        logger.info('directory: %s already exists, backing up this folder ... ' % directory)
        backup_dir = directory + '_backup'

        if os.path.exists(backup_dir):
            logger.info('backup directory also exists, removing the backup directory first')
            shutil.rmtree(backup_dir, True)

        shutil.copytree(directory, backup_dir)

    shutil.rmtree(directory, True)
    os.makedirs(directory)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.__sum = 0
        self.__count = 0

    def update(self, val, n=1):
        self.val = val
        self.__sum += val * n
        self.__count += n

    @property
    def avg(self):
        if self.__count == 0:
            return 0.
        return self.__sum / self.__count
