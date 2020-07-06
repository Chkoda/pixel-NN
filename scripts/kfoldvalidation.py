import logging
import os
import sys
import argparse
from collections import namedtuple
import numpy as np
import pandas as pd
import h5py as h5
import matplotlib.pyplot as plt
import tensorflow
import tensorflow.keras as keras
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

from run_training import _build_model, _find_py_file

#os.chdir('C:/Users/alexc/OneDrive/Projects/pixel-NN-master')
#sys.path.append('scripts')
#sys.path.append('share')
#sys.path.append('python')
#logging.getLogger('matplotlib.font_manager').disabled = True

def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--input', required=True)
    args.add_argument('--model', required=True)
    args.add_argument('--name')
    args.add_argument('--folds', default=10)
    args.add_argument('--epochs', required=True)
    args.add_argument('--loglevel', default='INFO')
    return args.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=1, format='[%(asctime)s %(levelname)s] %(message)s')
    Option = namedtuple("MyStruct", "input model name folds epochs")
    # structure learning_rate regularizer epochs")
    args = Option(
        input='training1.h5',
        model='share/reference_number.py',
        name='NumberNetworkKFOLD',
        folds=10,
        epochs=20,
        # learning_rate=0.0001,
        # regularizer=0.0001,
    )
    path = _find_py_file(args.model)