#!/bin/bash -l
#SBATCH -J LYAP			# Job name
#SBATCH -N 1            # Number of nodes to request
#SBATCH -n 35			# set the max number of tasks/cores allowed to launch per node
#SBATCH --time=96:00:00 # The time limit to give job


# load modules
module load fftw/3.3.8
module load gcc
module load gsl/2.5
module load hdf5/1.10.5
