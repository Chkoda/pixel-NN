#!/bin/bash
#SBATCH --account=atlas
#SBATCH --license=cfs
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J runtest
#SBATCH --mail-user=chkoda@uw.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:00:03


#set up environment
module load python/3.7-anaconda-2019.10
module load tensorflow/intel-2.2.0-py37

#run the application:
srun -n 1 -c 64 --cpu_bind=cores sh /global/cfs/cdirs/atlas/achkodrov/pixel-NN-master/slurm_run/train_numbernetwork.sh