#!/bin/env python2
""" run_training.py: driver script to train a neural network """

import argparse
import importlib as imp
import logging
import os
import h5py as h5
import tensorflow.keras as keras
import numpy as np

from sklearn.model_selection import KFold

keras.backend.set_floatx('float32')

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

    '''
    # --------Training---------
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

    logging.info('Fitting model')
    fit_args['verbose'] = 2
    history = model.fit(data_x, data_y, **fit_args)

    hpath = name + '.history.h5'
    logging.info('Writing fit history to %s', hpath)
    with h5.File(hpath, 'w') as hfile:
        for key, val in history.history.items():
            hfile.create_dataset(key, data=np.array(val))
    # ------------------------------
    '''

    # ------K-Fold Cross Validation------------
    from sklearn.model_selection import KFold
    models = []
    preds = []
    histories = []
    acc_per_fold = []
    loss_per_fold = []
    k_folds = 10
    kfold = KFold(n_splits=k_folds)

    #dont forget to shuffle
    data = np.concatenate([data_x, data_y], axis=1)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(data)
    data_x = data[:, 0:data_x.shape[1]]
    data_y = data[:, data_x.shape[1]:data.shape[1]]

    compile_args['metrics']=['accuracy']
    fit_args['epochs'] = 50
    fit_args['callbacks'] = []
    fit_args['validation_split'] = 0.0

    fold_no = 1
    for train, test in kfold.split(data_x, data_y):
        model, compile_args, fit_args, params = _build_model(path, data_x, data_y)
        fit_args.setdefault('callbacks', [])
        fit_args['callbacks'] += [keras.callbacks.TerminateOnNaN(),
                                  keras.callbacks.ModelCheckpoint(name + '.h5',save_best_only=True,verbose=1)]
        model.compile(**compile_args)

        print(f'Training for fold {fold_no} ...')
        history = model.fit(data_x[train], data_y[train], **fit_args)
        scores = model.evaluate(data_x[test], data_y[test])
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1]*100)
        loss_per_fold.append(scores[0])
        #hpath = name + '_fold_' + str(fold_no) + '.history.h5'
        #logging.info('Writing fit history to %s', hpath)
        #with h5.File(hpath, 'w') as hfile:
        #    for key, val in history.history.items():
        #        hfile.create_dataset(key, data=np.array(val))

        y_pred = model.predict(data_x[test])
        preds.append(y_pred)

        models.append(model)
        histories.append(history)
        fold_no += 1

    print('---Average scores---')
    print('Score per fold')
    for i in range(0,len(acc_per_fold)):
        print(f'Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('--------------------')
    print('Average scores for all folds:')
    print(f'Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'Loss: {np.mean(loss_per_fold)} (+- {np.std(loss_per_fold)})')

    # ---------------------------------------



if __name__ == '__main__':
    _main()
