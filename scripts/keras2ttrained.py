#!/bin/env python2

import argparse

import ROOT

from PixelNN import ttrained

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('input')
    args.add_argument('output')
    args = args.parse_args()

    ofile = ROOT.TFile.Open(args.output, 'CREATE')
    if ofile.IsZombie():
        return 1
    ttnet = ttrained.from_keras(keras.models.load_model(args.input))
    ttnet.Write(0, ROOT.TObject.kWriteDelete)
