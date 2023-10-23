default:
	gcc -O3 -fopenmp gof-parallel-bp.c

clean: 
	rm -f Bleona* Natalia* *.sh.e* *.sh.o*
