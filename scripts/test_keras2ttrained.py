#!/bin/env python2

import argparse

import keras
import numpy as np
import root_numpy
import ROOT

from PixelNN import keras_utils, ttrained, utils

_type_trans = {
    'number': 'NumberParticles',
    'pos1': 'ImpactPoints1P',
    'pos2': 'ImpactPoints2P',
    'pos3': 'ImpactPoints3P',
    'error1x': 'ImpactPointErrorsX1',
    'error2x': 'ImpactPointErrorsX2',
    'error3x': 'ImpactPointErrorsX3',
    'error1y': 'ImpactPointErrorsY1',
    'error2y': 'ImpactPointErrorsY2',
    'error3y': 'ImpactPointErrorsY3',
}

def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        'type',
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
        ]
    )
    args.add_argument('keras_path')
    args.add_argument('data_path')
    args.add_argument('--nclusters', type=int, default=1000)
    return args.parse_args()

def _main():
    args = _get_args()

    kmodel = keras.models.load_model(
        args.keras_path,
        custom_objects={
            'OffsetAndScale': keras_utils.OffsetAndScale,
            '_sigmoid2': keras_utils._sigmoid2
        }
    )
    tmodel = ttrained.from_keras(kmodel)
    kmodel2 = ttrained.to_keras(tmodel)

    xarray = root_numpy.root2array(
        args.data_path,
        treename='NNinput',
        branches=utils.get_branches(args.type)[0],
        start=0,
        stop=args.nclusters
    )
    xarray = xarray.view(np.float64).reshape(xarray.shape + (-1,))

    for i in range(xarray.shape[0]):
        x = xarray[i]
        xvec = ROOT.vector('Double_t')()
        for xval in x:
            xvec.push_back(xval)

        ky = kmodel.predict(x.reshape((1, x.shape[0])))[0]
        ty = np.array(tmodel.calculateNormalized(xvec))
        k2y = kmodel2.predict(x.reshape((1, x.shape[0])))[0]

        np.testing.assert_allclose(ky, ty, rtol=1e-2)
        np.testing.assert_allclose(ky, k2y, rtol=1e-2)


if __name__ == '__main__':
    _main()
