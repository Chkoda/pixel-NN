#!/bin/env python2
import argparse
import logging
import os
os.environ['KERAS_BACKEND'] = 'theano'

#import tensorflow as tf
#import tensorflow.keras as keras

import keras
import imp

import numpy as np
import h5py as h5

import sys
sys.path.append('python')

import keras_utils
from keras_utils import OffsetAndScale, _sigmoid2
#from run_training import _build_model

logging.basicConfig(
    level='INFO',
    format='[%(asctime)s %(levelname)s %(module)s.%(funcName)s] %(message)s'
)

def _build_model(path, data_x, data_y):
    """ Load a model from a .py source """
    #defm = SourceFileLoader('model_def', path).load_module()
    defm = imp.load_source('model_def', path)
    if 'build_model' not in dir(defm):
        raise RuntimeError("build_model() function not defined in '%s'" % path)
    buildf = defm.build_model
    return buildf(data_x, data_y)

def _apply_model(path, nntype, data_x, data_y):
    logging.info('Loading %s model from %s', nntype, path)

    model = keras.models.load_model(
        path,
        custom_objects={
            name: getattr(keras_utils, name) for name in dir(keras_utils)
        }
    )

    model_json = model.to_json()
    with open("modelWeights/LGconfig.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('modelWeights/LGmodelweights.h5')

    #model, compile_args, _, _ = _build_model("share/reference_number.py", data_x, data_y)
    #model.compile(**compile_args)
    #model.load_weights(path)

    logging.info('Fetching input data for %s', nntype)
    #x_branches, _ = utils.get_branches(nntype)
    nndata = data_x

    logging.info('Applying model to data')
    pred = model.predict(
        nndata
    )

    return pred


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--input', required=True)
    args.add_argument('--type', required=True, choices=['number', 'pos1', 'pos2', 'pos3'])
    args.add_argument('--output')
    args.add_argument('--name', default='apply_outdata')
    args.add_argument('--ITk', action='store_true')
    args.add_argument('--pitchX', type=float, default=0.05)
    args.add_argument('--pitchY', type=float, default=None) # None == variable
    args.add_argument('--number-thresholds', type=float, nargs=2, default=[0.6, 0.2])
    args.add_argument('--max-clusters', type=int, default=None)

    grp = args.add_mutually_exclusive_group(required=True)
    grp.add_argument('--model')
    grp.add_argument('--evaluated')
    grp.add_argument('--models', nargs=3)

    return args.parse_args()


def _main():

    args = _get_args()

    logging.info('Loading data from %s', args.input)
    with h5.File('data/total.h5', 'r') as data:
        data_x = data['input'][0:100]
        data_y = data['target'][0:100]
    
    #data = root_numpy.root2array(args.input, stop=args.max_clusters)

    if args.model:
        models = [args.model]
        types = [args.type]
    elif args.models:
        models = args.models
        types = 'pos{n} error{n}x error{n}y'.format(n=args.type[-1]).split()

    preds = []
    for model, nntype in zip(models, types):
        preds.append(_apply_model(model, nntype, data_x, data_y))

    if args.output:
        outpath = args.output
    else:
        outpath = os.path.basename(args.input) + '.NNapplied'
    logging.info('Output path: %s', outpath)

    if args.type == 'number':
        logging.info('Starting number NN validation results production')
        outdata = _do_number(data_x, data_y, preds[0], args.number_thresholds)
    elif args.type.startswith('pos'):
        logging.info('Starting position NN validation results production')
        if args.models:
            logging.info('Including error NNs results')
            errors = preds[1:]
        else:
            errors = None
        outdata = _do_pos(data, preds[0], int(args.type[-1]), args.pitchX, args.pitchY, errors)
        outpath += '.tmp'

    logging.info('Saving results to %s', outpath)

    with h5.File('{}/{}.h5'.format(outpath, args.name), 'w') as hfile:
        for key in outdata.dtype.names:
            hfile.create_dataset(
                key,
                data=outdata[key],
                compression='gzip'
            )
    #root_numpy.array2root(
    #    outdata,
    #    outpath,
    #    'NNoutput',
    #    'recreate'
    #)

    if args.type.startswith('pos'):
        logging.info(
            'Producing pos validation histograms, saving to %s',
             outpath.replace('.tmp', '')
        )
        npart = int(args.type[-1])
        validation.make_histograms(
            [outpath],
            outpath.replace('.tmp', ''),
            npart,
            IBL=(not args.ITk),
            pixdims=(
                args.pitchX,
                args.pitchY if args.pitchY else 0.5
            ),
            do_pulls=args.models is not None
        )
        os.remove(outpath)


################################################################################
### number NN


def _do_number(data_x, data_y, pred, thrs):

    outdata = np.zeros(
        (pred.shape[0],),
        dtype=[
            ('Output_number', np.dtype(('f4', 3))),
            ('Output_number_true', 'i4'),
            ('Output_number_estimated', 'i4'),
            ('NN_layer', 'i4'),
            ('NN_barrelEC', 'i4'),
            ('globalEta', 'f4'),
            ('globalPhi', 'f4'),
            ('cluster_size', 'i4'),
            ('cluster_size_X', 'i4'),
            ('cluster_size_Y', 'i4'),
        ]
    )

    outdata['Output_number'] = pred / np.sum(pred, axis=1, keepdims=True)
    outdata['Output_number_true'] = 1 * data_y[:,0] + \
                                    2 * data_y[:,1] + \
                                    3 * data_y[:,2]

    outdata['Output_number_estimated'][
        np.where(np.logical_and(pred[:,1] < thrs[0], pred[:,2] < thrs[1]))
    ] = 1
    outdata['Output_number_estimated'][
        np.where(np.logical_and(pred[:,1] >= thrs[0], pred[:,2] < thrs[1]))
    ] = 2
    outdata['Output_number_estimated'][
        np.where(pred[:,2] >= thrs[1])
    ] = 3

    auxfields = [
        'NN_layer',
        'NN_barrelEC',
        'globalEta',
        'globalPhi',
        'cluster_size',
        'cluster_size_X',
        'cluster_size_Y'
    ]

    outdata['NN_layer'] = data_x[:, 56]
    outdata['NN_barrelEC'] = data_x[:, 57]
    #for field in auxfields:
    #    outdata[field] = data_x[field]

    return outdata


################################################################################
### position NNs


def _error_rms(data, half_interval):

    vmin, vmax = -half_interval, half_interval
    indices = np.arange(data.shape[1])

    centers = np.tile(
        vmin + (vmax - vmin) * (indices - 0.5) / (indices.shape[0] - 2),
        (data.shape[0], 1)
    ) ** 2

    rms = np.sqrt(
        np.average(
            centers,
            weights=data,
            axis=1,
        )
    ).reshape((centers.shape[0], 1))

    centers = np.ma.array(
        centers,
        mask=np.logical_or(
            centers < - 3 * rms,
            centers > 3 * rms
        )
    )

    return np.sqrt(np.ma.average(centers, weights=data, axis=1))

def _do_pos(data, pred, npart, pitchX, pitchY, errors=None):

    dtype = [
            ('Output_positions_X', np.dtype((np.float32, npart))),
            ('Output_positions_Y', np.dtype((np.float32, npart))),
            ('Output_true_X', np.dtype((np.float32, npart))),
            ('Output_true_Y', np.dtype((np.float32, npart))),
            ('Output_number_true', 'i4'),
            ('NN_layer', 'i4'),
            ('NN_barrelEC', 'i4'),
            ('globalEta', 'f4'),
            ('globalPhi', 'f4'),
            ('cluster_size', 'i4'),
            ('cluster_size_X', 'i4'),
            ('cluster_size_Y', 'i4'),
    ]

    if errors:
        dtype += [
            ('Output_uncert_X', np.dtype((np.float32, npart))),
            ('Output_uncert_Y', np.dtype((np.float32, npart)))
        ]

    outdata = np.zeros(
        (pred.shape[0],),
        dtype=dtype
    )

    outdata['Output_positions_X'] = pred[:, [i*2 for i in range(npart)]].reshape(
        outdata['Output_positions_X'].shape
    )
    outdata['Output_positions_Y'] = pred[:, [i*2+1 for i in range(npart)]].reshape(
        outdata['Output_positions_Y'].shape
    )

    if errors:
        if npart == 1:
            maxx = 0.04
            maxy = 0.3
            nbins = 30
            outdata['Output_uncert_X'] = _error_rms(errors[0][:, :nbins], maxx)
            outdata['Output_uncert_Y'] = _error_rms(errors[1][:, :nbins], maxy)
        else:

            if npart == 2:
                maxx = 0.05
                maxy = 0.4
                nbins = 25

            else:
                maxx = 0.05
                maxy = 0.4
                nbins = 20

            for i in range(npart):
                outdata['Output_uncert_X'][:, i] = _error_rms(
                    errors[0][:, i*nbins:(i+1)*nbins],
                    maxx
                )
                outdata['Output_uncert_Y'][:, i] = _error_rms(
                    errors[1][:, i*nbins:(i+1)*nbins],
                    maxy
                )

    if npart == 1:
        outdata['Output_true_X'][:,np.newaxis][:,i] = data['NN_position_id_X_{}'.format(i)]
        outdata['Output_true_Y'][:,np.newaxis][:,i] = data['NN_position_id_Y_{}'.format(i)]
    else:
        for i in range(npart):
            outdata['Output_true_X'][:,i] = data['NN_position_id_X_{}'.format(i)]
            outdata['Output_true_Y'][:,i] = data['NN_position_id_Y_{}'.format(i)]


    outdata['Output_number_true'] = npart

    outdata['Output_positions_X'] *= pitchX
    outdata['Output_true_X'] *= pitchX

    pitches = np.stack(
        [data['NN_pitches{}'.format(i)] for i in range(7)],
        axis=-1
    )

    def _conv_Y(offsets):

        if len(offsets.shape) == 1:
            offsets = offsets.reshape((offsets.shape[0], 1))

        # `offsets' are relative to center so `indices' here is
        # relative to matrix origin
        indices = 3.5 + offsets
        # the integer part identifies the containing pixel
        int_indices = np.floor(indices).astype(int, copy=False)
        # the fractional part identifies the position within the
        # containing pixel
        frac_indices = indices - int_indices

        # here, just clip everything to the matrix
        # TODO: change this to account for hit outside matrix
        toolow = np.where(int_indices < 0)
        toohigh = np.where(int_indices > 6)
        frac_indices[toolow] = 0.0
        frac_indices[toohigh] = 1.0
        int_indices = np.clip(int_indices, 0, 6)

        # the cumulative sum at index `i' is the distance relative to
        # the matrix origin when position is just past pixel `i'. Here
        # a column of 0s is prepended such that this quantity at index
        # i is the distance relative to the origin when position is
        # just before pixel 'i', so we can use int_indices directly
        sums = np.concatenate(
            [np.zeros((pitches.shape[0],1)), np.cumsum(pitches, axis=1)],
            axis=1
        )
        total = np.empty_like(int_indices, dtype=np.float64)
        allrows = np.arange(offsets.shape[0])

        for i in range(npart):
            idx = int_indices[:, i]
            total[:, i] = pitches[allrows, idx]
            total[:, i] *= frac_indices[allrows, i]
            total[:, i] += sums[allrows, idx]

        return total

    if pitchY is None:
        outdata['Output_positions_Y'] = _conv_Y(outdata['Output_positions_Y']).reshape(
            outdata['Output_positions_Y'].shape
        )
        outdata['Output_true_Y'] = _conv_Y(outdata['Output_true_Y']).reshape(
            outdata['Output_true_Y'].shape
        )
    else:
        outdata['Output_positions_Y'] *= pitchY
        outdata['Output_true_Y'] *= pitchY

    auxfields = [
        'NN_layer',
        'NN_barrelEC',
        'globalEta',
        'globalPhi',
        'cluster_size',
        'cluster_size_X',
        'cluster_size_Y'
    ]

    for field in auxfields:
        outdata[field] = data[field]


    return outdata


if __name__ == '__main__':
    _main()
