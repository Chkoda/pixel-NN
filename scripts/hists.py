#!/bin/env python2

import argparse
import importlib

args = argparse.ArgumentParser()
args.add_argument('--input', required=True)
args.add_argument('--output', required=True)
args = args.parse_args()

validation = importlib.import_module('.validation', 'pixel-NN')
validation.make_histograms(args.input, args.output)
