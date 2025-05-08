#!/bin/bash -l

PROJECT_DIR=$HOME/Codes/graph-consensus-clustering
DATA_DIR=$PROJECT_DIR/test/data
RESULT_DIR=$PROJECT_DIR/test/experiment-results

export OMP_NUM_THREADS=8
export OMP_PLACES=cores

BIN=$PROJECT_DIR/preprocess

OUTPUT_DIR_NAME=preprocess-study
OUTPUT_DIR="$RESULT_DIR"/"$OUTPUT_DIR_NAME"
mkdir -p $OUTPUT_DIR

echo $OUTPUT_DIR

#for DATASET_NAME in LFR-louvain
#for DATASET_NAME in LFR-louvain-1 LFR-louvain-2 LFR-louvain-3 LFR-louvain-4 LFR-louvain-5 LFR-louvain-6 LFR-louvain-7 LFR-louvain-8 LFR-louvain-9 LFR-louvain-10
for DATASET_NAME in LFR-1 LFR-2 LFR-3 LFR-4 LFR-5 LFR-6 LFR-7 LFR-8 LFR-9 LFR-10
do
    #for N in 200 1000 5000
    for N in 5000
    do
        for MU in 01 02 03 04 05 06 07
        #for MU in 07
        do
            STDOUT_FILE=$OUTPUT_DIR/"$DATASET_NAME".n"$N".mu"$MU".csv
            if [ -f $STDOUT_FILE ]; then
                rm -rf $STDOUT_FILE
            fi

            #echo $STDOUT_FILE

            echo "threshold,k,n_components" >> $STDOUT_FILE

            INPUT_CLUSTERING_PREFIX=$DATA_DIR/$DATASET_NAME/n$N/LFR_n"$N"_mu"$MU"_gamma30_beta11

            NUMBER_OF_INPUT_CLUSTERING=0
            while [ $NUMBER_OF_INPUT_CLUSTERING -ne 100 ]
            do
                CLUSTER_FILE="$INPUT_CLUSTERING_PREFIX"."$NUMBER_OF_INPUT_CLUSTERING"

                if [ -f $CLUSTER_FILE ]; then
                    NUMBER_OF_INPUT_CLUSTERING=$(($NUMBER_OF_INPUT_CLUSTERING+1))
                else
                    break;
                fi
            done

            echo mu, $MU
            for T in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
            do
                STDOUT=$($BIN --input-prefix "$INPUT_CLUSTERING_PREFIX" --n "$N" --k "$NUMBER_OF_INPUT_CLUSTERING" --pre-proc-threshold "$T")
                echo $STDOUT
                echo $STDOUT >> $STDOUT_FILE
            done
        done
    done
done

