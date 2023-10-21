default:
	gcc -O3 -fopenmp -DOUTPUT gof-parallel.c

clean: 
	rm -f Bleona* Natalia* *.sh.e* *.sh.o*
