#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J runtraining
#SBATCH --account=atlas
#SBATCH --mail-user=chkoda@uw.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00