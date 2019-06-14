"""
author     : Mert Bulent Sariyildiz
email      : mbsariyildiz@gmail.com
last update: 11-04-2019
"""

import os
import json
import numpy as np
import datetime
from collections import OrderedDict
from .misc import AverageMeter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

__all__ = ['Logger']

class Logger(object):

    def __init__(self, log_dir, label, titles):
        """
        log_dir      : str, directory where all the logs will be written.
        label        : str, root filename for the logs. It shouldn't contain an extension, such as .txt
        titles       : list, title for each log attribute.
        """

        self.log_dir = log_dir
        self.label = label
        self.titles = titles

        self.logs = {} # all title-log pairs that will be traced for this instance
        self.meters = {}
        for t in titles:
            self.logs[t] = []
            self.meters[t] = AverageMeter()

        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
        self.tb_logger = SummaryWriter(self.log_dir)

    def save_logs(self):
        """
        Saves raw log values in both numpy arrays and matplotlib plots.
        """
        self.save_as_arrays()
        self.save_as_figures()

    def close(self):
        """
        """
        self.save_logs()
        self.tb_logger.close()

    def update_meters(self, titles, values, n=1):
        """
        Updates average meter of each title in titles.
        If step is multiple of append_steps, then self.append is called.

        titles : list, entries must be in self.titles.
        values : list, must be of the same size as self.titles.
        n      : number of samples whose values are aggregated into values.
        """
        assert len(titles) == len(values)

        for t, v in zip(titles, values):
            self.meters[t].update(v, n)

    def flush_meters(self, step):
        """
        Calls self.append with meters whose average value is non-zero. The function also
        resets values of the meters.
        """

        titles = []
        values = []
        for t, m in self.meters.items():
            if m.avg != 0:
                titles.append(t)
                values.append(m.avg)
                m.reset()

        self.append(titles, values, step)

    def append(self, titles, values, step):
        """
        Adds a new log value for each title in titles.

        titles : list, entries must be in self.titles.
        values : list, value for each title in titles.
        step   : int, a step number for log summary
        """
        if titles is None: titles = self.titles
        assert len(titles) == len(values)

        step_log = OrderedDict()
        step_log['step'] = str(step)
        step_log['time'] = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")

        for t, v in zip(titles, values):
            self.logs[t].append(v)
            step_log[t] = v
            self.tb_logger.add_scalar(t, v, step)

        f_txt = open(os.path.join(self.log_dir, '{}.txt'.format(self.label)), 'a')
        json.dump(step_log, f_txt, indent=4)
        f_txt.write('\n')
        f_txt.flush()
        f_txt.close()

    def save_as_arrays(self):
        """
        Converts all logs to numpy arrays and saves them into self.log_dir.
        """
        arrays = {}
        for t, v in self.logs.items():
            if len(v) > 0:
                v = np.array(v)
                arrays[t] = v

        if len(arrays) > 0:
            np.savez(
                os.path.join(self.log_dir, '{}.npz'.format(self.label)), **arrays)

    def save_as_figures(self):
        """
        First, converts all logs to numpy arrays, then plots them using matplotlib. Finally, saves the plots into self.log_dir.
        """
        for t, v in self.logs.items():
            if len(v) > 0:
                v = np.array(v)

                fig = plt.figure(dpi=400)
                ax = fig.add_subplot(111)
                ax.plot(v)
                ax.set_title(t)
                ax.grid(True)
                fig.savefig(
                    os.path.join(self.log_dir, '{}_{}.png'.format(self.label, t.replace('/', '_'))),
                    bbox_inches='tight' )
                plt.close()

