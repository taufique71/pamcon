PROJECT_DIR = $(HOME)/Codes/graph-consensus-clustering
NIST_DIR = $(PROJECT_DIR)/NIST

CC = g++ -fopenmp -fpermissive -std=c++11 -O3 

mmio.o: $(NIST_DIR)/mmio.c
	$(CC)  -c -o mmio.o $(NIST_DIR)/mmio.c

main.o: main.cpp
	$(CC) -c -o main.o main.cpp

v8-cpp: main.o mmio.o
	$(CC) -o v8-cpp main.o mmio.o

all: v8-cpp

clean:
	rm -f mmio.o
	rm -f main.o
	rm -f v8-cpp
