CC=mpicc
CFLAGS=-O3
RES=-DPRINT_RESULTS
CONV=-DTEST_CONV

all: jacobi jacobi-conv

jacobi: Jacobi_mpi.c utils.c
	$(CC) $(CFLAGS) $(RES) -o $@ $^

jacobi-conv: Jacobi_mpi.c utils.c
	$(CC) $(CFLAGS) $(RES) $(CONV) -o $@ $^

cleandist:
	rm -f jacobi jacobi-conv

clean:
	rm -f *.err *.out
