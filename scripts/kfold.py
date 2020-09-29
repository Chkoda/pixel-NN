import logging
import argparse
import sys
import math
import numpy as np
import h5py as h5
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

#import tensorflow
#import tensorflow.keras as keras
import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
import imp

keras.backend.set_floatx('float32')

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
    args.add_argument('--folds', default=10)
    args.add_argument('--batch_size', default=60)
    args.add_argument('--epochs', default=1000)
    args.add_argument('--select_folds', nargs='*', default=-1)
    return args.parse_args()

def _main():
    args = _get_args()
    path = _find_py_file(args.model)

    # Load and Split Data
    logging.info('Loading data from {}'.format(args.input))

    with h5.File(args.input, 'r') as data:
<<<<<<< HEAD
        data_x = data['input'][()]
        data_y = data['target'][()]
        #data_x = data['input'][0:5000]
        #data_y = data['target'][0:5000]
=======
        #data_x = data['input'][()]
        #data_y = data['target'][()]
        data_x = data['input'][0:5000]
        data_y = data['target'][0:5000]
>>>>>>> master


    #DATA COMES PRESHUFFLED NOW
    #data = np.concatenate([data_x, data_y], axis=1)
    #rng = np.random.default_rng(seed=42)
    #rng.shuffle(data)
    #data_x = data[:, 0:data_x.shape[1]]
    #data_y = data[:, data_x.shape[1]:data.shape[1]]

    kfold = KFold(n_splits=args.folds)
    splits = kfold.split(data_x,data_y)
    fold_no = 0


    if type(args.select_folds) == list: # != -1:
        logging.info('Custom kfold range')
        if len(args.select_folds) == 2:
            sel_folds = [int(args.select_folds[0]), int(args.select_folds[1])]
            fold_no = int(args.select_folds[0])
            logging.info('From kfold {} to {}'.format(sel_folds[0], sel_folds[1]))
            splits = [(train, test) for train, test in splits][sel_folds[0]:sel_folds[1]]
        elif len(args.select_folds) == 1:
<<<<<<< HEAD
            sel_fold = int(args.select_folds[0]) - 1
            fold_no = sel_fold
            logging.info('kfold {}'.format(sel_fold))
            splits = [[(train, test) for train, test in splits][sel_fold]]
=======
            sel_fold = int(args.select_folds[0])
            fold_no = sel_fold
            logging.info('kfold {}'.format(sel_fold))
            splits = [(train, test) for train, test in splits][sel_fold]
>>>>>>> master
        else:
            raise Exception('Too many command line arguments for select folds!')
    
    # Training and Validation

    _, compile_args, fit_args, _ = _build_model(path, data_x, data_y)
    compile_args['metrics']=['accuracy']
    fit_args['batch_size']=args.batch_size
    fit_args['validation_split'] = 0.1
    fit_args['epochs'] = args.epochs
    
    
    with h5.File('output/{}.history.h5'.format(args.name), 'w'):
        logging.info('output/{}.history.h5 created/emptied'.format(args.name))

    for train, test in splits:
        fold_no += 1
        logging.info('Building model for fold {} from {}...'.format(fold_no, path))
        model, _, _, _ = _build_model(path, data_x, data_y)

        logging.info('Compiling for fold {} ...'.format(fold_no))
        model.compile(**compile_args)

        logging.info('Training for fold {} ...'.format(fold_no))
        fit_args['callbacks'] += [keras.callbacks.TerminateOnNaN(), 
                                keras.callbacks.ModelCheckpoint(
                                    'output/' + args.name + str(fold_no) + '.h5', 
                                    save_best_only=True,
                                    save_weights_only=False,
                                    verbose=1)
                                ]
        history = model.fit(data_x[train], data_y[train], **fit_args)
        
        logging.info('Writing fit history to output/{}.history.h5'.format(args.name))
        with h5.File('output/{}.history.h5'.format(args.name), 'a') as hfile:
            for key, val in history.history.items():
                hfile.create_dataset(key+'_'+str(fold_no), data=np.array(val))

    logging.info('Done training!')

if __name__ == '__main__':
    _main()
