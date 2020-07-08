#!/bin/env python2
""" run_training.py: driver script to train a neural network """

import argparse
from importlib.machinery import SourceFileLoader
import logging
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import h5py as h5
import tensorflow.keras as keras
import numpy as np
import multiprocessing

from keras.callbacks import TensorBoard
keras.backend.set_floatx('float32')


import datetime

def _find_py_file(path):

    if os.path.exists(path):
        return path

    try:
        pathlist = os.environ['DATAPATH']
        for dirpath in pathlist.split(':'):
            newpath = '{}/{}'.format(dirpath, path)
            if os.path.exists(newpath):
                return newpath
    except KeyError:
        pass

    raise IOError('File not found: %s' % path)


def _build_model(path, data_x, data_y):
    """ Load a model from a .py source """
    defm = SourceFileLoader('model_def', path).load_module()
    if 'build_model' not in dir(defm):
        raise RuntimeError("build_model() function not defined in '%s'" % path)
    buildf = defm.build_model
    return buildf(data_x, data_y)


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--input', required=True)
    args.add_argument('--model', required=True)
    args.add_argument('--name')
    args.add_argument('--loglevel', default='INFO')
    return args.parse_args()


def _main():

    args = _get_args()

    logging.basicConfig(
        level=args.loglevel,
        format='[%(asctime)s %(levelname)s] %(message)s'
    )

    if args.name is None:
        name = os.path.basename(args.model).replace('.py', '')
    else:
        name = args.name

    logging.info('Loading data from %s', args.input)
    with h5.File(args.input, 'r') as hfile:
        data_x = hfile['input'][()]
        data_y = hfile['target'][()]

    path = _find_py_file(args.model)
    logging.info('Building model from %s', path)
    model, compile_args, fit_args, params = _build_model(path, data_x, data_y)

    if params:
        ppath = name + '.parameters.h5'
        logging.info('Writing parameters to %s', ppath)
        with open(ppath, 'w') as pfile:
            pfile.write(repr(params))

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    fit_args.setdefault('callbacks', [])
    
    fit_args['callbacks'] += [
        tensorboard_callback,
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(
            name + '.h5',
            save_best_only=True,
            verbose=2
        )
    ]
    fit_args['use_multiprocessing'] = True
    fit_args['workers'] = int(multiprocessing.cpu_count()*0.9)
    logging.info(f'Fit with cpu count: {fit_args["workers"]}')

    logging.info('Compiling model')
    model.compile(**compile_args)

    logging.info('Fitting model')
    fit_args['verbose'] = 2
    history = model.fit(data_x, data_y, **fit_args)

    hpath = name + '.history.h5'
    logging.info('Writing fit history to %s', hpath)
    with h5.File(hpath, 'w') as hfile:
        for key, val in history.history.items():
            hfile.create_dataset(key, data=np.array(val))


if __name__ == '__main__':
    _main()
