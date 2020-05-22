#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N oscillators
#$ -pe mvapich2-grotrian 16
#$ -t 1:48
#$ -o oeFiles/o$JOB_NAME_$TASK_ID.$JOB_ID.dat
#$ -e oeFiles/e$JOB_NAME_$TASK_ID.$JOB_ID.dat
echo "got $NSLOTS slots."
export OMP_NUM_THREADS=16
echo "Start time is `date`"
./main.exe $SGE_TASK_ID 
echo "End time is `date`"
exit 0
