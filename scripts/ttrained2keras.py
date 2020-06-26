#!/bin/env python2

import argparse
import os

import ROOT

from PixelNN import ttrained

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('input')
    args.add_argument('treename')
    args.add_argument('output')
    args = args.parse_args()

    if os.path.exists(args.output):
        raise RuntimeError('output file {} exists!'.format(args.output))

    ifile = ROOT.TFile.Open(args.input, 'READ')
    ttnet = ifile.Get(args.treename)
    knet = ttrained.to_keras(ttnet)
    knet.save(args.output)
