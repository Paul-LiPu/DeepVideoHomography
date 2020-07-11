import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime
import collections
import _pickle as pickle
import time
import logging
import sys

import os

class Metric_Logger():
    """
    Class to remember training metrics and output as files or images.
    """
    def __init__(self, logdir='.', logger=None, visual=False, tick_interval=1):
        self._since_beginning = collections.defaultdict(lambda: {})
        self._since_last_flush = collections.defaultdict(lambda: {})
        self.logdir = logdir
        self.logfile = self.logdir + '/log.pkl'
        self.logger = logger
        self.visual = visual
        self._iter = 0
        self.first_time = time.time()
        self.time = time.time()

    def log(self, name, value, iter):
        self._iter = iter
        self._since_last_flush[name][iter] = value

    def flush(self):
        prints = []

        for name, vals in self._since_last_flush.items():
            prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
            self._since_beginning[name].update(vals)

            # temp = self._since_beginning[name].keys()
            x_vals = np.sort(list(self._since_beginning[name].keys()))
            y_vals = [self._since_beginning[name][x] for x in x_vals]

            plt.clf()
            plt.plot(x_vals, y_vals)
            plt.xlabel('iteration')
            plt.ylabel(name)

            plt.savefig(self.logdir + '/' + name.replace(' ', '_') + '.jpg')

        curr_time = time.time()
        elapsed_time = curr_time - self.first_time
        interval_time = curr_time - self.time
        if self.logger is None:
            print("iter {}\t{}".format(self._iter, "\t".join(prints)))
        else:
            self.logger.info("[iter {} | {} | {}]\t{}".format(self._iter, str(datetime.timedelta(seconds=elapsed_time)), str(elapsed_time) \
                                                          + 's', str(interval_time) + 's',"\t".join(prints)))
        self._since_last_flush.clear()

        self.time = curr_time
        with open(self.logfile, 'wb') as f:
            pickle.dump(dict(self._since_beginning), f)

    def resume(self):
        with open(self.logfile, 'rb') as f:
            while True:
                try:
                    obj = pickle.load(f)
                    for key in obj:
                        self._since_beginning[key] = obj[key]
                except EOFError:
                    break



def getLogger(logFile, fmt='%(asctime)s %(message)s', datefmt='[%Y/%m/%d %H:%M:%S]', style='%'):
    # Get a logger
    logger = logging.getLogger(os.path.basename(logFile))

    # level
    logger.setLevel(logging.DEBUG)

    # formatter
    formatter = logging.Formatter(fmt, datefmt, style)

    # handlers
    stdout_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(logFile, 'w')
    handlers = [stdout_handler, file_handler]

    # apply handler and formatter
    for h in handlers:
        if h.formatter is None:
            h.setFormatter(formatter)
        logger.addHandler(h)

    return logger