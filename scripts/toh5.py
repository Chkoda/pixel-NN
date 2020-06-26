#!/bin/env python2

import argparse
import os

import h5py as h5
import root_numpy
import numpy as np

from PixelNN import utils

def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--input', required=True)
    args.add_argument('--output')
    args.add_argument('--nclusters', type=int)
    args.add_argument(
        '--type',
        choices=[
            'number',
            'pos1',
            'pos2',
            'pos3',
            'error1x',
            'error1y',
            'error2x',
            'error2y',
            'error3x',
            'error3y',
        ],
        required=True
    )
    return args.parse_args()


def _add(rootfile, name, branches, h5file, stop):
    array = root_numpy.root2array(
        filenames=rootfile,
        branches=branches,
        stop=stop
    )
    array = array.view(np.float64).reshape(array.shape + (-1,))
    h5file.create_dataset(
        name,
        data=array.astype(np.float32),
        compression='gzip'
    )


def _main():
    args = _get_args()

    x_branches, y_branches = utils.get_branches(args.type)

    if args.output is None:
        output = os.path.basename(args.input).replace('.root', '') + '.h5'
    else:
        output = args.output

    with h5.File(output, 'w') as hfile:
        _add(args.input, 'input', x_branches, hfile, args.nclusters)
        _add(args.input, 'target', y_branches, hfile, args.nclusters)


if __name__ == '__main__':
    _main()
