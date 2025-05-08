#!/bin/bash -l

PROJECT_DIR=$HOME/Codes/graph-consensus-clustering
DATA_DIR=/home/mth/Data/UNC_DATASET/Metis_Format
RESULT_DIR=$PROJECT_DIR/test/experiment-results/fastpg

#export OMP_NUM_THREADS=8
#export OMP_PLACES=cores

BIN=$PROJECT_DIR/consensus
KIRKLEY_NEWMAN_SCRIPT=$PROJECT_DIR/Kirkley-Newman/main.py
LANCICHINETTI_FORTUNATO_SCRIPT=$PROJECT_DIR/Lancichinetti-Fortunato/main.py
STDOUT_PREFIX=$PROJECT_DIR/stdout.FASTPG

#for ALG in v8 v8-parallel
#for ALG in v8-parallel
for ALG in kirkley-newman
do
    #for DATA_NAME in Levine13_dimNetworkMetis Levine32_dimNetworkMetis Samusik_01NetworkMetis Samusik_allNetworkMetis
    for DATA_NAME in Samusik_allNetworkMetis
    #for DATA_NAME in Samusik_01NetworkMetis
    #for DATA_NAME in Levine13_dimNetworkMetis 
    #for DATA_NAME in Levine32_dimNetworkMetis 
    do
        OUTPUT_DIR_NAME="$ALG"."$DATA_NAME"
        OUTPUT_DIR="$RESULT_DIR"/"$OUTPUT_DIR_NAME"
        if [ -d $OUTPUT_DIR ]; then
            rm -rf $OUTPUT_DIR
        fi
        mkdir -p $OUTPUT_DIR

        STDOUT_FILE=$RESULT_DIR/stdout."$OUTPUT_DIR_NAME"
        if [ -f $STDOUT_FILE ]; then
            rm -rf $STDOUT_FILE
        fi

        echo $STDOUT_FILE

        GRAPH_FILE=$DATA_DIR/$DATA_NAME.mtx
        INPUT_CLUSTERING_PREFIX=$DATA_DIR/$DATA_NAME
        OUTPUT_CLUSTERING_PREFIX=$OUTPUT_DIR/$DATA_NAME

        NUMBER_OF_INPUT_CLUSTERING=0
        while [ $NUMBER_OF_INPUT_CLUSTERING -ne 100 ]
        do
            CLUSTER_FILE="$INPUT_CLUSTERING_PREFIX"."$NUMBER_OF_INPUT_CLUSTERING"
            #echo $CLUSTER_FILE

            if [ -f "$CLUSTER_FILE" ]; then
                NUMBER_OF_INPUT_CLUSTERING=$(($NUMBER_OF_INPUT_CLUSTERING+1))
            else
                break;
            fi
        done

        #$BIN --graph-file "$GRAPH_FILE" \
        #--input-prefix "$INPUT_CLUSTERING_PREFIX" \
        #--k "$NUMBER_OF_INPUT_CLUSTERING" \
        #--output-prefix "$OUTPUT_CLUSTERING_PREFIX" \
        #--pre-proc-threshold "$T" \
        #--alg $ALG >> "$STDOUT_FILE"."$TRD"
        
        #echo $BIN
        #echo $GRAPH_FILE
        #echo $INPUT_CLUSTERING_PREFIX
        #echo $NUMBER_OF_INPUT_CLUSTERING
        #echo $T
        #echo $ALG

        #for TRD in 128 64 16 4 1
        for TRD in 1
        do
            echo ::: Running with $TRD threads :::
            export OMP_NUM_THREADS=$TRD
            export OMP_PLACES=cores
            if [ "$ALG" == "kirkley-newman" ]; then
                source /home/mth/.venv/bin/activate
                python $KIRKLEY_NEWMAN_SCRIPT --graph-file "$GRAPH_FILE" \
                --input-prefix "$INPUT_CLUSTERING_PREFIX" \
                --k "$NUMBER_OF_INPUT_CLUSTERING" \
                --output-prefix "$OUTPUT_CLUSTERING_PREFIX" \
                --alg $ALG >> "$STDOUT_FILE"
                deactivate
            elif [ "$ALG" == "v8-parallel" ]; then
                T=0.99
                $BIN --graph-file "$GRAPH_FILE" \
                --input-prefix "$INPUT_CLUSTERING_PREFIX" \
                --k "$NUMBER_OF_INPUT_CLUSTERING" \
                --output-prefix "$OUTPUT_CLUSTERING_PREFIX" \
                --pre-proc-threshold "$T" \
                --alg $ALG >> "$STDOUT_FILE"."$TRD"
            fi
        done
    done
done
