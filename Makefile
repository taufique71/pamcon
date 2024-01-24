LOCAL_HOME_DIR = /home/mth
PROJECT_DIR = $(HOME)/Codes/graph-consensus-clustering
NIST_DIR = $(PROJECT_DIR)/NIST
CONPAS_DIR = $(PROJECT_DIR)/CONPAS
IGRAPH_DIR = $(LOCAL_HOME_DIR)/Software/igraph-0.10.6/install

CFLAGS = -std=c++11 -O3 -fopenmp -fpermissive -ffast-math -fomit-frame-pointer -msse -mmmx  
#CFLAGS = -std=c++11 -g -O0 -fopenmp -fpermissive -ffast-math -fomit-frame-pointer -msse -mmmx  
CC = g++ 

AdjBestOfK.o : $(CONPAS_DIR)/AdjBestOfK.cpp $(CONPAS_DIR)/AdjBestOfK.h $(CONPAS_DIR)/CCHeuristic.h
	$(CC) $(CFLAGS) -c -o AdjBestOfK.o $(CONPAS_DIR)/AdjBestOfK.cpp

AverageLink.o : $(CONPAS_DIR)/AverageLink.h $(CONPAS_DIR)/AverageLink.cpp
	$(CC) $(CFLAGS) -c -o AverageLink.o $(CONPAS_DIR)/AverageLink.cpp

BestOfK.o : $(CONPAS_DIR)/BestOfK.cpp $(CONPAS_DIR)/BestOfK.h $(CONPAS_DIR)/MajorityRule.h $(CONPAS_DIR)/CCHeuristic.h
	$(CC) $(CFLAGS) -c -o BestOfK.o $(CONPAS_DIR)/BestOfK.cpp

CCOptimal.o : $(CONPAS_DIR)/CCOptimal.cpp $(CONPAS_DIR)/CCOptimal.h $(CONPAS_DIR)/CCHeuristic.h
	$(CC) $(CFLAGS) -c -o CCOptimal.o $(CONPAS_DIR)/CCOptimal.cpp

CCAverageLink.o : $(CONPAS_DIR)/CCAverageLink.cpp $(CONPAS_DIR)/CCAverageLink.h 
	$(CC) $(CFLAGS) -c -o CCAverageLink.o $(CONPAS_DIR)/CCAverageLink.cpp

MajorityRule.o : $(CONPAS_DIR)/MajorityRule.h $(CONPAS_DIR)/MajorityRule.cpp $(CONPAS_DIR)/SetPartition.h $(CONPAS_DIR)/CCHeuristic.h $(CONPAS_DIR)/SetPartitionVector.h
	$(CC) $(CFLAGS) -c -o MajorityRule.o $(CONPAS_DIR)/MajorityRule.cpp

MersenneTwister.o : $(CONPAS_DIR)/MersenneTwister.cpp $(CONPAS_DIR)/MersenneTwister.h
	$(CC) $(CFLAGS) -c -o MersenneTwister.o $(CONPAS_DIR)/MersenneTwister.cpp

SetPartition.o : $(CONPAS_DIR)/SetPartition.h $(CONPAS_DIR)/SetPartition.cpp $(CONPAS_DIR)/Utility.h $(CONPAS_DIR)/MersenneTwister.h
	$(CC) $(CFLAGS) -c -o SetPartition.o $(CONPAS_DIR)/SetPartition.cpp

SetPartitionVector.o : $(CONPAS_DIR)/SetPartitionVector.cpp $(CONPAS_DIR)/SetPartitionVector.h $(CONPAS_DIR)/SetPartition.h $(CONPAS_DIR)/Utility.h
	$(CC) $(CFLAGS) -c -o SetPartitionVector.o $(CONPAS_DIR)/SetPartitionVector.cpp

Utility.o : $(CONPAS_DIR)/Utility.h $(CONPAS_DIR)/Utility.cpp
	$(CC) $(CFLAGS) -c -o Utility.o $(CONPAS_DIR)/Utility.cpp

CCPivot.o : $(CONPAS_DIR)/CCPivot.h $(CONPAS_DIR)/CCPivot.cpp
	$(CC) $(CFLAGS) -c -o CCPivot.o $(CONPAS_DIR)/CCPivot.cpp

RefinedClustering.o : $(CONPAS_DIR)/RefinedClustering.h $(CONPAS_DIR)/RefinedClustering.cpp
	$(CC) $(CFLAGS) -c -o RefinedClustering.o $(CONPAS_DIR)/RefinedClustering.cpp

mmio.o: $(NIST_DIR)/mmio.c
	$(CC) $(CFLAGS) -c -o mmio.o $(NIST_DIR)/mmio.c

consensus.o: consensus.cpp
	$(CC) $(CFLAGS) -c -o consensus.o consensus.cpp

consensus: consensus.o mmio.o AdjBestOfK.o AverageLink.o BestOfK.o CCOptimal.o MajorityRule.o MersenneTwister.o SetPartition.o SetPartitionVector.o Utility.o CCPivot.o CCAverageLink.o RefinedClustering.o
	$(CC) $(CFLAGS) -o consensus *.o

dist-distribution: dist-distribution.cpp
	$(CC) $(CFLAGS) dist-distribution.cpp \
		-I$(IGRAPH_DIR)/include/igraph \
		-L$(IGRAPH_DIR)/lib \
		-ligraph -lm -larpack \
		-o dist-distribution

all: consensus dist-distribution

clean:
	rm -f *.o
	rm -f consensus
	rm -f dist-distribution 
