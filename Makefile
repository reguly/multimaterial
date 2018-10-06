CC=g++ #pgc++
ACCFLAGS=-DACC -DLINKED -fast -O3 -acc -ta=tesla,cc60 -Minfo=acc -mp -Mcuda #-Ofast -mavx2 -mp
OMPFLAGS=-DOMP -O0 -g -fopenmp #-mp -fastsse -fast #-DLINKED
NVCCFLAGS=-O3 -arch=sm_60 #-DFUSED #-DLINKED

.cpp.o:
	$(CC) -c $(CFLAGS) $<

all: multimat_acc multimat_omp

clean:
	rm -f *.o
	rm -f multimat_*

multimat_acc:
	$(CC) $(ACCFLAGS) -c compact.cpp -o compact.o
	$(CC) $(ACCFLAGS) -c full_matrix.cpp -o full_matrix.o
	$(CC) $(ACCFLAGS) -c multimat.cpp -o multimat.o
	$(CC) $(ACCFLAGS) -o $@ compact.o full_matrix.o multimat.o -lm

multimat_omp:
	$(CC) $(OMPFLAGS) -c compact.cpp -o compact.o
	$(CC) $(OMPFLAGS) -c full_matrix.cpp -o full_matrix.o
	$(CC) $(OMPFLAGS) -c multimat.cpp -o multimat.o
	$(CC) $(OMPFLAGS) -o $@ compact.o full_matrix.o multimat.o -lm

multimat_cuda:
	nvcc $(NVCCFLAGS) -c compact.cu -o compact.o
	nvcc $(NVCCFLAGS) -c full_matrix.cu -o full_matrix.o
	$(CC) $(OMPFLAGS) -c multimat.cpp -o multimat.o
	$(CC) $(ACCFLAGS) -o $@ compact.o full_matrix.o multimat.o -lm
