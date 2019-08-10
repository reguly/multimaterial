#CC=g++ #pgc++
#ACCFLAGS=-DACC -O3 -fast -g -acc -ta=tesla,cc60 -Minfo=acc -mp -Mcuda #-Ofast -mavx2 -mp
#OMPFLAGS=-DOMP -O3 -g -fopenmp #-DFUSED -DLINKED #-fopenmp #-mp -fastsse -fast #-DLINKED
#NVCCFLAGS=-O0 -g -G -arch=sm_60 #-DFUSED -DLINKED

#CC=icpc #pgc++
#ACCFLAGS=-DACC -O3 -fast -g -acc -ta=tesla,cc60 -Minfo=acc -mp -Mcuda #-Ofast -mavx2 -mp
#OMPFLAGS=-DOMP -O3 -g -qopenmp -qopt-report=5 -xHost -fp-model=fast #-DFUSED -DLINKED #-fopenmp #-mp -fastsse -fast #-DLINKED
#NVCCFLAGS=-O3 -g -arch=sm_60 #-DFUSED -DLINKED

ifeq ($(CC),g++)
	OMPFLAGS=-DOMP -Ofast -ffp-contract=fast -fopenmp
	AFF=OMP_PROC_BIND=TRUE
	NVCCFLAGS=-O3 -use_fast_math -arch=sm_60 -Xcompiler=-fopenmp
endif
ifeq ($(CC),icpc)
	OMPFLAGS=-DOMP -O3 -g -qopenmp -qopt-report=5 -xHost -fp-model=fast
	AFF=KMP_AFFINITY=compact
	NVCCFLAGS=-O3 -use_fast_math -arch=sm_60
endif
ifeq ($(CC),pgc++)
	OMPFLAGS=-DOMP -Ofast -fast -mp
	NVCCFLAGS=-O3 -use_fast_math -arch=sm_60
	ACCFLAGS=-DACC -O3 -fast -acc -ta=tesla,cc60 -Minfo=acc -mp -Mcuda #-Ofast -mavx2 -mp
endif

.cpp.o:
	$(CC) -c $(CFLAGS) $<

all: multimat_acc multimat_omp

clean:
	rm -f *.o
	rm -f multimat_*

multimat_acc_FL:
	$(CC) $(ACCFLAGS) -DFUSED -DLINKED -c compact.cpp -o compact.o
	$(CC) $(OMPFLAGS) -DFUSED -DLINKED -c full_matrix.cpp -o full_matrix.o
	$(CC) $(ACCFLAGS) -DFUSED -DLINKED -c multimat.cpp -o multimat.o
	$(CC) $(ACCFLAGS) -o $@ compact.o full_matrix.o multimat.o -lm

multimat_omp_FL:
	$(CC) $(OMPFLAGS) -DFUSED -DLINKED -c compact.cpp -o compact.o
	$(CC) $(OMPFLAGS) -DFUSED -DLINKED -c full_matrix.cpp -o full_matrix.o
	$(CC) $(OMPFLAGS) -DFUSED -DLINKED -c multimat.cpp -o multimat.o
	$(CC) $(OMPFLAGS) -o $@ compact.o full_matrix.o multimat.o -lm

multimat_cuda_FL:
	nvcc $(NVCCFLAGS) -DFUSED -DLINKED -c compact.cu -o compact.o
	#nvcc $(NVCCFLAGS) -DFUSED -DLINKED -c full_matrix.cu -o full_matrix.o
	$(CC) $(OMPFLAGS) -DFUSED -DLINKED -c full_matrix.cpp -o full_matrix.o
	$(CC) $(OMPFLAGS) -DFUSED -DLINKED -c multimat.cpp -o multimat.o
	nvcc $(NVCCFLAGS) -o $@ compact.o full_matrix.o multimat.o -lm

multimat_acc_F:
	$(CC) $(ACCFLAGS) -DFUSED -c compact.cpp -o compact.o
	$(CC) $(OMPFLAGS) -DFUSED -c full_matrix.cpp -o full_matrix.o
	$(CC) $(ACCFLAGS) -DFUSED -c multimat.cpp -o multimat.o
	$(CC) $(ACCFLAGS) -o $@ compact.o full_matrix.o multimat.o -lm

multimat_omp_F:
	$(CC) $(OMPFLAGS) -DFUSED -c compact.cpp -o compact.o
	$(CC) $(OMPFLAGS) -DFUSED -c full_matrix.cpp -o full_matrix.o
	$(CC) $(OMPFLAGS) -DFUSED -c multimat.cpp -o multimat.o
	$(CC) $(OMPFLAGS) -o $@ compact.o full_matrix.o multimat.o -lm

multimat_cuda_F:
	nvcc $(NVCCFLAGS) -DFUSED -c compact.cu -o compact.o
	#nvcc $(NVCCFLAGS) -DFUSED -c full_matrix.cu -o full_matrix.o
	$(CC) $(OMPFLAGS) -DFUSED -c full_matrix.cpp -o full_matrix.o
	$(CC) $(OMPFLAGS) -DFUSED -c multimat.cpp -o multimat.o
	nvcc $(NVCCFLAGS) -o $@ compact.o full_matrix.o multimat.o -lm

multimat_acc:
	$(CC) $(ACCFLAGS) -c compact.cpp -o compact.o
	$(CC) $(OMPFLAGS) -c full_matrix.cpp -o full_matrix.o
	$(CC) $(ACCFLAGS) -c multimat.cpp -o multimat.o
	$(CC) $(ACCFLAGS) -o $@ compact.o full_matrix.o multimat.o -lm

multimat_omp:
	$(CC) $(OMPFLAGS) -c compact.cpp -o compact.o
	$(CC) $(OMPFLAGS) -c full_matrix.cpp -o full_matrix.o
	$(CC) $(OMPFLAGS) -c multimat.cpp -o multimat.o
	$(CC) $(OMPFLAGS) -o $@ compact.o full_matrix.o multimat.o -lm

multimat_cuda:
	nvcc $(NVCCFLAGS) -c compact.cu -o compact.o
	#nvcc $(NVCCFLAGS) -c full_matrix.cu -o full_matrix.o
	$(CC) $(OMPFLAGS) -c full_matrix.cpp -o full_matrix.o
	$(CC) $(OMPFLAGS) -c multimat.cpp -o multimat.o
	nvcc $(NVCCFLAGS) -o $@ compact.o full_matrix.o multimat.o -lm

test_cpu: multimat_omp multimat_omp_F multimat_omp_FL
	$(AFF) ./multimat_omp_FL 3000 3000
	$(AFF) ./multimat_omp_F 3000 3000
	$(AFF) ./multimat_omp 3000 3000
	$(AFF) ./multimat_omp_FL 3000 3000 0.3 0.05 0.05
	$(AFF) ./multimat_omp_F 3000 3000 0.3 0.05 0.05
	$(AFF) ./multimat_omp 3000 3000 0.3 0.05 0.05

test_gpu: multimat_cuda multimat_cuda_F multimat_cuda_FL
	$(AFF) ./multimat_cuda_FL 3000 3000
	$(AFF) ./multimat_cuda_F 3000 3000
	$(AFF) ./multimat_cuda 3000 3000
	$(AFF) ./multimat_cuda_FL 3000 3000 0.3 0.05 0.05
	$(AFF) ./multimat_cuda_F 3000 3000 0.3 0.05 0.05
	$(AFF) ./multimat_cuda 3000 3000 0.3 0.05 0.05

test_acc: multimat_acc multimat_acc_F multimat_acc_FL
	$(AFF) ./multimat_acc_FL 3000 3000
	$(AFF) ./multimat_acc_F 3000 3000
	$(AFF) ./multimat_acc 3000 3000
	$(AFF) ./multimat_acc_FL 3000 3000 0.3 0.05 0.05
	$(AFF) ./multimat_acc_F 3000 3000 0.3 0.05 0.05
	$(AFF) ./multimat_acc 3000 3000 0.3 0.05 0.05
