#!/bin/env python2
""" run_training.py: driver script to train a neural network """

import argparse
import imp
import logging
import os

import h5py as h5
import keras
import numpy as np

from sklearn.model_selection import KFold

keras.backend.set_floatx('float64')

import sys
sys.path.append('share')

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
    defm = imp.load_source('model_def', path)
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
    data = h5.File(args.input, 'r')
    data_x = data['input'][()]
    data_y = data['target'][()]

    path = _find_py_file(args.model)
    logging.info('Building model from %s', path)
    model, compile_args, fit_args, params = _build_model(path, data_x, data_y)

    if params:
        ppath = name + '.parameters.h5'
        logging.info('Writing parameters to %s', ppath)
        with open(ppath, 'w') as pfile:
            pfile.write(repr(params))

    fit_args.setdefault('callbacks', [])
    fit_args['callbacks'] += [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(
            name + '.h5',
            save_best_only=True,
            verbose=1
        )
    ]

    logging.info('Compiling model')
    model.compile(**compile_args)

    # ------K-Fold Cross Validation------------

    # ---------------------------------------

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
