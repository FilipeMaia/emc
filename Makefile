all: emc

emc.o: emc.c
	gcc -g -std=c99 -c emc.c `gsl-config --cflags` -I/usr/common/usg/cuda/3.2/include/

emc_cuda.o: emc_cuda.cu
#	nvcc -m64 -c emc_cuda.cu 
	nvcc -m64 -c emc_cuda.cu

emc_atomic.o: emc_atomic.cu
	nvcc -m64 -c emc_atomic.cu -arch sm_20 -G -g

emc: emc.o emc_cuda.o emc_atomic.o
	nvcc -o emc emc.o emc_cuda.o emc_atomic.o -m64 -lspimage `gsl-config --libs` -L/usr/local/cuda/lib

clean:
	rm *.o emc
