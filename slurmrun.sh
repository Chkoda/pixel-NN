#!/bin/bash
#SBATCH -A m1759
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 01:45:00
#SBATCH -J runtest
#SBATCH --mail-user=chkoda@uw.edu
#SBATCH --mail-type=ALL


#set up environment
module load gcc
module load cmake
module load esslurm

conda activate py2theano

#run the application:
srun python /global/cfs/cdirs/atlas/achkodrov/pixel-NN/scripts/run_training.py --input /global/cfs/cdirs/atlas/achkodrov/data/train.h5 --model /global/cfs/cdirs/atlas/achkodrov/pixel-NN/share/reference_number.py --name /global/cfs/cdirs/atlas/achkodrov/pixel-NN/output/TrainedNN_7_6
