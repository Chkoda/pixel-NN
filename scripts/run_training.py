#!/bin/env python2
""" run_training.py: driver script to train a neural network """

import argparse
import logging
import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
np.random.seed(42)
import multiprocessing
import h5py as h5


#from importlib.machinery import SourceFileLoader
#import tensorflow as tf
#import tensorflow.keras as keras
#from tensorflow.keras.callbacks import TensorBoard

import imp
import keras

from keras.utils import plot_model

keras.backend.set_floatx('float32')


import datetime

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
'''

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
    #defm = SourceFileLoader('model_def', path).load_module()
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
    with h5.File(args.input, 'r') as hfile:
        # data_x = hfile['input'][0:1000]
        # data_y = hfile['target'][0:1000]
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

    #log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "\\"
    #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    fit_args.setdefault('callbacks', [])
    
    fit_args['callbacks'] += [
        #tensorboard_callback,
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(
            name,
            save_best_only=True,
            verbose=2
        )
    ]
    #fit_args['use_multiprocessing'] = True
    #fit_args['workers'] = int(multiprocessing.cpu_count()*0.9)
    #logging.info(f'Fit with cpu count: {fit_args["workers"]}')

    logging.info('Compiling model')
    model.compile(**compile_args)
    #plot_model(model, to_file='model.png', show_shapes=True)
    logging.info('Fitting model')
    fit_args['verbose'] = 2

    #trainDataSet = tf.data.Dataset.from_tensor_slices((data_x, data_y)).batch(60)
    #history = model.fit(trainDataSet, **fit_args)
    
    history = model.fit(data_x, data_y, **fit_args)
    model.save(name + '_final.h5')
    '''
    model.save('lwnntest.h5')
    model.save_weights('lwtnnweights.h5')
    with open("lwtnnjson.json", 'w') as ofile:
        ofile.write(model.to_json())
    '''
    

    hpath = name + '.history.h5'
    logging.info('Writing fit history to %s', hpath)


    with h5.File(hpath, 'w') as hfile:
        for key, val in history.history.items():
            hfile.create_dataset(key, data=np.array(val))


if __name__ == '__main__':
    _main()
