#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_fw

## Output and error files
#PBS -o result.out
#PBS -e error.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=1

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab19/pps/a2/FW
export OMP_NUM_THREADS=64
for x in 64 128 256; do
	./fw_sr 2048 $x
done
