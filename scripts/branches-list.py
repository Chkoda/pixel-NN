#!/bin/env python2

import argparse

argp = argparse.ArgumentParser()
argp.add_argument("-i", "--ignore", nargs='*', default=[])
argp.add_argument("-f", "--add-float", nargs='*', default=[])
argp.add_argument("-d", "--add-int", nargs='*', default=[])
argp.add_argument("-x", "--xdim", type=int, default=7)
argp.add_argument("-y", "--ydim", type=int, default=7)
argp.add_argument("type", choices=['number', 'pos1', 'pos2', 'pos3'])
args = argp.parse_args()

branches = []
types = []
for i in range(args.xdim * args.ydim):
    branches.append('NN_matrix%d' % i)
    types.append('float')

for i in range(args.ydim):
    branches.append('NN_pitches%d' % i)
    types.append('float')

branches.append('NN_layer')
types.append('int')
branches.append('NN_barrelEC')
types.append('int')
branches.append('NN_phi')
types.append('float')
branches.append('NN_theta')
types.append('float')

for br in args.add_float:
    branches.append(br)
    types.append('float')
for br in args.add_int:
    branches.append(br)
    types.append('int')

if args.type == 'number':
    branches.append('NN_nparticles1')
    types.append('int')
    branches.append('NN_nparticles2')
    types.append('int')
    branches.append('NN_nparticles3')
    types.append('int')

if args.type in ['pos1', 'pos2', 'pos3']:
    branches.append('NN_position_id_X_0')
    types.append('float')
    branches.append('NN_position_id_Y_0')
    types.append('float')
if args.type in ['pos2', 'pos3']:
    branches.append('NN_position_id_X_1')
    types.append('float')
    branches.append('NN_position_id_Y_1')
    types.append('float')
if args.type in ['pos3']:
    branches.append('NN_position_id_X_2')
    types.append('float')
    branches.append('NN_position_id_Y_2')
    types.append('float')

for br in args.ignore:
    try:
        idx = branches.index(br)
        branches.pop(idx)
        types.pop(idx)
    except ValueError:
        pass

def prn(str, indent):
    print (' ' * indent) + str

prn('branches:', 0)
for i in range(len(branches)):
    prn('%s: %s' % (branches[i], types[i]), 2)
prn('order:', 0)
for i in range(len(branches)):
    prn('- %s' % branches[i], 2)
