#!/bin/bash -l
#SBATCH -J LYAP			# Job name
#SBATCH -N 1            # Number of nodes to request
#SBATCH -n 35			# set the max number of tasks/cores allowed to launch per node
#SBATCH -A ucd01        # The account to charge compute time to
#SBATCH --time=96:00:00 # The time limit to give job
#SBATCH -p LongQ        # Which queue to place job in

# Load modules
module load fftw/3.3.8_double_i18u4
module load hdf5/intel/1.10.1
module load gsl/intel/2.5
source ~/ecpy/bin/activate

cd /ichec/home/users/endacarroll/Burgers/burgers-code/Lyapunov_Analysis/LCE_Computation/

python3 run_lce_in_parallel.py