#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_kmeans

## Output and error files
#PBS -o resut.out
#PBS -e error.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=8

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab19/pps/a2/kmeans
export OMP_NUM_THREADS=4
## export GOMP_CPU_AFFINITY="0-63:2"
## ./kmeans_omp_critical -s 256 -n 16 -c 16 -l 10
## ./kmeans_omp_reduction -s 256 -n 16 -c 16 -l 10
## ./kmeans_omp_reduction -s 256 -n 1 -c 4 -l 10
./kmeans_omp_reduction_ft -s 256 -n 1 -c 4 -l 10
