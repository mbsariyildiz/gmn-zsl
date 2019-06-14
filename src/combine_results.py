"""
author     : Mert Bulent Sariyildiz
email      : mbsariyildiz@gmail.com
last update: 11-04-2019
"""

import os
import argparse
import glob
import numpy as np
import utils 

parser = argparse.ArgumentParser()
parser.add_argument('--glob_root', type=str,
                    help='directory which includes all experiment folders')
parser.add_argument('--log_file', type=str,
                    help='the path of a log file relative to glob_root')
parser.add_argument('--save_dir', type=str,
                    help='the path of the folder where combined logs will be written')
parser.add_argument('--keys', type=str,
                    help='comma separated names of the attributes saved in log file')
args = parser.parse_args()

def main():

    folders = glob.glob(os.path.join(args.glob_root, '*'))
    folders = [f for f in folders if os.path.isdir(f) and 'combined-results' not in f]
    print ('Experiment folders under {}:'.format(args.glob_root))
    for folder in folders: print ('\t{}'.format(folder))

    keys = args.keys.split(',') # names of the log attributes in log files
    print ('Attributes whose values will be combined:', keys)
    max_key_len = max([len(k) for k in keys])
    combined_logs = {}
    for k in keys: combined_logs[k] = []

    utils.prepare_directory(args.save_dir, force_delete=True)
    logger = utils.Logger(
        args.save_dir, 
        'results', 
        list(map(lambda k: '{}/mean'.format(k), keys)) + \
        list(map(lambda k: '{}/std'.format(k), keys)))

    for folder in folders:
        
        log_file = os.path.join(folder, args.log_file) # log file in the experiment folder
        if not os.path.exists(log_file): continue

        print ('\tLoading log file {} ...'.format(log_file))
        L = np.load(log_file)
        for k in keys:
            v = L[k]
            print ('\t\tkey:{key:{width}s}, value.shape:{keyshape:15s}, value[-1]:{keyval:}'.format(
                key=k, width=max_key_len+1, keyshape=str(np.asarray(v).shape), keyval=v[-1]))
            if len(v.shape) < 2:
                v = v.reshape([-1, 1])
            combined_logs[k].append(v)

    for k in keys:
        combined_k = np.concatenate(combined_logs[k], axis=1)
        mean_k = combined_k.mean(axis=1)
        std_k = combined_k.std(axis=1)
        for lix, (mean, std) in enumerate(zip(mean_k, std_k)):
            logger.append(['{}/mean'.format(k), '{}/std'.format(k)], [mean, std], lix)
        
        print ('\taverage {key:{width}s}: {mean:} +- {std:}'.format(
            key=k, width=max_key_len+1, mean=mean_k[-1], std=std_k[-1]))

    logger.close()

if __name__ == '__main__':
    main()
