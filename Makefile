all: emc

emc.o: emc.c
	gcc -O2 -std=c99 -c emc.c

emc_cuda.o: emc_cuda.cu
	nvcc -m64 -c emc_cuda.cu

emc: emc.o emc_cuda.o
	nvcc -o emc emc.o emc_cuda.o -m64 -lspimage -lgsl -L/usr/local/cuda/lib

clean:
	rm *.o emc
