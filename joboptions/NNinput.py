import argparse
import logging
import os
import random

import AthenaRootComps.ReadAthenaxAODHybrid

# Get the user options
args = argparse.ArgumentParser()
args.add_argument('--output')
args.add_argument('--type')
args.add_argument('--max-clusters', type=int, default=17000000)
args.add_argument('--fractions', type=float, nargs=3, default=[0.22, 0.26, 0.52])
args.add_argument('--seed', type=int, default=random.randint(2,1000000))
args.add_argument('--ttrained')
args.add_argument('--max-eta', type=float, default=2.5)
args.add_argument('--max-index', type=float, default=float('inf'))
args = args.parse_args(opts.user_opts)


jobname = os.path.basename(script)
logging.info("Using joboption: %s", jobname)

if args.type is None:
    typename = None
    for script in opts.scripts:
        if 'NNinput_' in script:
            typename  = script.replace('.py', '').split('_')[-1]
            break
    if typename is None:
        raise RuntimeError("no 'NNinput_*' joboption specified")
else:
    typename = args.type

logging.info("NN type: %s", typename)

algSeq = CfgMgr.AthSequencer("AthAlgSeq")
alg = CfgMgr.ClustersLoop()

alg.max_eta = args.max_eta
alg.max_index = args.max_index

try:
    alg.NNtype = {
        'number': 0,
        'pos1': 1,
        'pos2': 2,
        'pos3': 3
    }[typename]
except KeyError:
    raise RuntimeError("Unknown NN type: %s", typename)

if typename == 'number':
    alg.nclusters_1 = int(args.max_clusters * args.fractions[0])
    alg.nclusters_2 = int(args.max_clusters * args.fractions[1])
    alg.nclusters_3 = int(args.max_clusters * args.fractions[2])
    alg.shuffle = args.seed
elif typename == 'pos1':
    alg.nclusters_1 = args.max_clusters
    alg.nclusters_2 = 0
    alg.nclusters_3 = 0
elif typename == 'pos2':
    alg.nclusters_1 = 0
    alg.nclusters_2 = args.max_clusters
    alg.nclusters_3 = 0
elif typename == 'pos3':
    alg.nclusters_1 = 0
    alg.nclusters_2 = 0
    alg.nclusters_3 = args.max_clusters
    alg.nclusters_3_inclusive = False

if typename != 'number' and args.ttrained is not None:
    alg.TTrainedNetworks_path = args.ttrained
    logging.info('TTrainedNetworks available from %s', args.ttrained)

algSeq += alg

if args.output is None:
    outpath = "NNinput_{}.root".format(typename)
else:
    outpath = args.output

logging.info('Output path: %s', outpath)
svcMgr.THistSvc.Output += [
    "OUTTREE DATAFILE='{}' OPT='RECREATE'".format(outpath)
]
