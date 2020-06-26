#!/bin/env python2

import argparse
import importlib
import logging

logging.basicConfig(level=logging.INFO)

args = argparse.ArgumentParser()
args.add_argument('input')
args.add_argument('output')
args = args.parse_args()

validation = importlib.import_module('.validation', 'pixel-NN')
validation.make_histograms([args.input], args.output)
