#!/bin/sh

## Give the Job a descriptive name
#PBS -N make_jacobi_mpi

## Output and error files
#PBS -o make.out
#PBS -e make.err

## How many machines should we get?
#PBS -l nodes=1:ppn=1

#PBS -l walltime=00:02:00

## Start
## Run make in the src folder (modify properly)

module load openmpi/1.8.3
cd /home/parallel/parlab19/newpps/a4/heat_transfer/mpi
make
