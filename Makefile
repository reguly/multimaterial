CC=g++
CFLAGS=-Ofast -mavx2 -ftree-vectorizer-verbose=1 -fopt-info-vec

.c.o:
	$(CC) -c $(CFLAGS) $<

all: multimat

clean:
	rm -f *.o
	rm -f multimat

multimat: compact.o full_matrix.o multimat.o
	$(CC) $(CFLAGS) -o $@ $^ -lm
