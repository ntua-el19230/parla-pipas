#!/bin/sh

## Give the Job a descriptive name
#PBS -N run_jacobi-noconv_mpi

## Output and error files
#PBS -o result.out
#PBS -e result.err

## How many machines should we get?
#PBS -l nodes=8:ppn=8

#PBS -l walltime=01:00:00

## Start
## Run make in the src folder (modify properly)

module load openmpi/1.8.3
cd /home/parallel/parlab19/newpps/a4/heat_transfer/mpi

for i in 2048 4096 6144
do
  mpi_run -np 1 --mca btl tcp,self ./jacobi-noconv $i $i 1 1
  mpi_run -np 2 --mca btl tcp,self ./jacobi-noconv $i $i 2 1
  mpi_run -np 4 --mca btl tcp,self ./jacobi-noconv $i $i 2 2
  mpi_run -np 8 --mca btl tcp,self ./jacobi-noconv $i $i 2 4
  mpi_run -np 16 --mca btl tcp,self ./jacobi-noconv $i $i 4 4
  mpi_run -np 32 --mca btl tcp,self ./jacobi-noconv $i $i 4 8
  mpi_run -np 64 --mca btl tcp,self ./jacobi-noconv $i $i 8 8
done

##mpi_run -np 64 --mca btl tcp,self ./jacobi-conv 1024 1024 8 8
