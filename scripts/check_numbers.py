#!/bin/env python2

import argparse
import logging
import numpy as np
import ROOT
import root_numpy as rnp

logging.basicConfig(level=logging.INFO)

p = argparse.ArgumentParser()
p.add_argument('input_file')
p.add_argument('--tree', default='NNinput')
args = p.parse_args()

logging.info('checking file: {}'.format(args.input_file))
logging.info('TTree: {}'.format(args.tree))

tfile = ROOT.TFile(args.input_file)
if tfile.IsZombie():
    logging.error('error while opening file')
    exit(1)

ttree = tfile.Get(args.tree)
if ttree == None:
    logging.error('error while retrieving TTree')
    exit(2)

p = 'P' if args.tree == 't1' else 'p'

arr = rnp.tree2array(ttree, branches=['NN_n{}articles1'.format(p), 'NN_n{}articles2'.format(p), 'NN_n{}articles3'.format(p)])


nclusters = arr.shape[0]
nparticles1 = np.sum(arr['NN_n{}articles1'.format(p)])
nparticles2 = np.sum(arr['NN_n{}articles2'.format(p)])
nparticles3 = np.sum(arr['NN_n{}articles3'.format(p)])


f1 = 100.0 * nparticles1 / nclusters
f2 = 100.0 * nparticles2 / nclusters
f3 = 100.0 * nparticles3 / nclusters

logging.info('total number of clusters: {}'.format(nclusters))
logging.info('1 particle clusters: {} ({}%)'.format(nparticles1, f1))
logging.info('2 particle clusters: {} ({}%)'.format(nparticles2, f2))
logging.info('3 particle clusters: {} ({}%)'.format(nparticles3, f3))
logging.info('sum of all types: {}'.format(nparticles1 + nparticles2 + nparticles3))

# nparts = np.sum(arr, axis=0)

# print nparts
