#!/bin/bash

cores=8
array_size=64

## Give the Job a descriptive name
#PBS -N Bleona

## Output and error files
#PBS -o Bleona.out
#PBS -e Bleona.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=8

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab19/pps/a1
export OMP_NUM_THREADS=8
./a.out 64 1000

