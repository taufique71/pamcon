#!/bin/bash -l

PROJECT_DIR=$HOME/Codes/graph-consensus-clustering
DATA_DIR=/home/mth/Data/UNC\ DATASET/Metis\ Format

export OMP_NUM_THREADS=8
export OMP_PLACES=cores

ALG=v8-parallel
BIN=$PROJECT_DIR/consensus
STDOUT_PREFIX=$PROJECT_DIR/stdout.FASTPG

#for ALG in v8 v8-parallel
for ALG in v8-parallel
do
    for DATA_NAME in Levine13_dimNetworkMetis Levine32_dimNetworkMetis Samusik_01NetworkMetis Samusik_allNetworkMetis
    #for DATA_NAME in Samusik_allNetworkMetis
    #for DATA_NAME in Levine13_dimNetworkMetis 
    do
        GRAPH_FILE=$DATA_DIR/$DATA_NAME.mtx
        INPUT_CLUSTERING_PREFIX=$DATA_DIR/$DATA_NAME.
        OUTPUT_CLUSTERING_FILE=$INPUT_CLUSTERING_PREFIX"$ALG"

        NUMBER_OF_INPUT_CLUSTERING=0
        while [ $NUMBER_OF_INPUT_CLUSTERING -ne 100 ]
        do
            CLUSTER_FILE=$INPUT_CLUSTERING_PREFIX"$NUMBER_OF_INPUT_CLUSTERING"
            #echo $CLUSTER_FILE

            if [ -f "$CLUSTER_FILE" ]; then
                NUMBER_OF_INPUT_CLUSTERING=$(($NUMBER_OF_INPUT_CLUSTERING+1))
            else
                break;
            fi
        done

        STDOUT_FILE="$STDOUT_PREFIX"."$ALG"."$DATA_NAME"
        if [ -f $STDOUT_FILE ]; then
            rm -rf $STDOUT_FILE
        fi

        echo "$STDOUT_FILE"

        $BIN --graph-file "$GRAPH_FILE" \
            --input-clustering-prefix "$INPUT_CLUSTERING_PREFIX" \
            --number-of-input-clustering "$NUMBER_OF_INPUT_CLUSTERING" \
            --output-file "$OUTPUT_CLUSTERING_FILE" \
            --alg "$ALG" >> "$STDOUT_FILE"
            #--alg "$ALG"
    done
done
