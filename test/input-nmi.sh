#!/bin/bash -l

PROJECT_DIR=$HOME/Codes/graph-consensus-clustering
DATA_DIR=$PROJECT_DIR/test/data
RESULT_DIR=$PROJECT_DIR/test/experiment-results

export OMP_NUM_THREADS=48
export OMP_PLACES=cores

BIN=$PROJECT_DIR/dist-distribution

#for DATASET_NAME in LFR-louvain LFR-mcl LFR
#for DATASET_NAME in LFR-preprocessed
for DATASET_NAME in LFR
do
    #for N in 200 1000 5000
    for N in 5000
    do
        for MU in 01 02 03 04 05 06 07
        #for MU in 01
        do
            FILE_PREFIX=LFR_n"$N"_mu"$MU"_gamma30_beta11
            OUTPUT_DIR_NAME="$ALG"."$DATASET_NAME".n"$N".mu"$MU"
            OUTPUT_DIR="$RESULT_DIR"/"$OUTPUT_DIR_NAME"
            INPUT_CLUSTERING_PREFIX=$DATA_DIR/$DATASET_NAME/n$N/$FILE_PREFIX
            OUTPUT_CLUSTERING_PREFIX=$OUTPUT_DIR/$FILE_PREFIX
            GT_FILE=$INPUT_CLUSTERING_PREFIX.gt

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

            for DIST in nmi
            do
                SOLN=0
                while [ $SOLN -lt $NUMBER_OF_INPUT_CLUSTERING ]
                do
                    #echo Solution: $SOLN
                    CONSENSUS_FILE="$INPUT_CLUSTERING_PREFIX"."$SOLN"
                    DISTANCE_DISTRIBUTION_FILE="$CONSENSUS_FILE".$DIST
                    if [ -f $CONSENSUS_FILE ]; then
                        $BIN --consensus-file $CONSENSUS_FILE \
                            --input-clustering-prefix $INPUT_CLUSTERING_PREFIX \
                            --number-of-input-clustering $NUMBER_OF_INPUT_CLUSTERING \
                            --ground-truth-file $GT_FILE \
                            --output-file $DISTANCE_DISTRIBUTION_FILE \
                            --distance-metric $DIST
                    else
                        break;
                    fi
                    echo ---
                    SOLN=$(($SOLN+1))
                done
            done
        done
    done
done
