#!/bin/env python2
import argparse
import logging

import keras
import ROOT

from PixelNN import keras_utils, ttrained


logging.basicConfig(
    level='INFO',
    format='[%(asctime)s %(levelname)s %(module)s.%(funcName)s] %(message)s'
)


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--number')
    args.add_argument('--pos1')
    args.add_argument('--pos2')
    args.add_argument('--pos3')
    args.add_argument('--error1x')
    args.add_argument('--error1y')
    args.add_argument('--error2x')
    args.add_argument('--error2y')
    args.add_argument('--error3x')
    args.add_argument('--error3y')
    args.add_argument('--output', required=True)
    return args.parse_args()


def _main():
    args = _get_args()

    logging.info('Creating output file %s', args.output)
    ofile = ROOT.TFile.Open(args.output, 'CREATE')
    if ofile.IsZombie():
        raise RuntimeError('Unable to create output file {}'.format(args.output))

    for key_k, key_tt in ttrained.nn_key_dict.items():
        if getattr(args, key_k):
            logging.info('Found %s network, saving as %s', key_k, key_tt)
            kmodel = keras_utils.load_model(getattr(args, key_k))
            ttmodel = ttrained.from_keras(kmodel)
            ttmodel.Write(key_tt, ROOT.TObject.kWriteDelete)

    logging.info('done')
    ofile.Close()


if __name__ == '__main__':
    _main()
