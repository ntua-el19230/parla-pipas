#!/bin/sh

## Give the Job a descriptive name
#PBS -N run_jacobi-conv_mpi

## Output and error files
#PBS -o result-conv.out
#PBS -e result-conv.err

## How many machines should we get?
#PBS -l nodes=8:ppn=8

#PBS -l walltime=01:00:00

## Start
## Run make in the src folder (modify properly)

module load openmpi/1.8.3
cd /home/parallel/parlab19/newpps/a4/heat_transfer/mpi
mpi_run -np 64 --mca btl tcp,self ./jacobi-conv 1024 1024 8 8
