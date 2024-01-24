#!/bin/bash -l

PROJECT_DIR=$HOME/Codes/graph-consensus-clustering
DATA_DIR=$PROJECT_DIR/test/data

export OMP_NUM_THREADS=48
export OMP_PLACES=cores

BIN=$PROJECT_DIR/dist-distribution
for DIST in vi split-join rand
#for DIST in rand
do
    #for ALG in v8-parallel v8 boem saoem
    for ALG in saoem
    do
        ##STDOUT_FILE=$PROJECT_DIR/stdout.LFR.$ALG

        #if [ -f $STDOUT_FILE ]; then
            #rm -rf $STDOUT_FILE
        #fi
        
        #echo $STDOUT_FILE

        #for N in 200 1000 5000
        for N in 5000
        do
            #for MU in 01 02 03 04
            for MU in 01
            do
                GRAPH_FILE=$DATA_DIR/LFR/n$N/LFR_n"$N"_mu"$MU"_gamma30_beta11.mtx
                INPUT_CLUSTERING_PREFIX=$DATA_DIR/LFR/n$N/LFR_n"$N"_mu"$MU"_gamma30_beta11.
                CONS_CLUSTERING_FILE=$INPUT_CLUSTERING_PREFIX"$ALG"
                DISTANCE_DISTRIBUTION_FILE=$INPUT_CLUSTERING_PREFIX"$ALG"."$DIST"

                NUMBER_OF_INPUT_CLUSTERING=0
                while [ $NUMBER_OF_INPUT_CLUSTERING -ne 100 ]
                do
                    CLUSTER_FILE=$INPUT_CLUSTERING_PREFIX"$NUMBER_OF_INPUT_CLUSTERING"

                    if [ -f $CLUSTER_FILE ]; then
                        NUMBER_OF_INPUT_CLUSTERING=$(($NUMBER_OF_INPUT_CLUSTERING+1))
                    else
                        break;
                    fi
                done

                $BIN --consensus-file $CONS_CLUSTERING_FILE \
                    --input-clustering-prefix $INPUT_CLUSTERING_PREFIX \
                    --number-of-input-clustering $NUMBER_OF_INPUT_CLUSTERING \
                    --output-file $DISTANCE_DISTRIBUTION_FILE \
                    --distance-metric $DIST
            done
        done
    done
done