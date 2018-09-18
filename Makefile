CC=g++
CFLAGS=-Ofast -mavx2 -fopenmp

.cpp.o:
	$(CC) -c $(CFLAGS) $<

all: multimat

clean:
	rm -f *.o
	rm -f multimat

multimat: compact.o full_matrix.o multimat.o
	$(CC) $(CFLAGS) -o $@ $^ -lm
