#!/bin/bash -l

PROJECT_DIR=$HOME/Codes/graph-consensus-clustering
DATA_DIR=$PROJECT_DIR/test/data
RESULT_DIR=$PROJECT_DIR/test/experiment-results/runtime-study

#export OMP_NUM_THREADS=8
#export OMP_PLACES=cores

BIN=$PROJECT_DIR/consensus
#KIRKLEY_NEWMAN_SCRIPT=$PROJECT_DIR/Kirkley-Newman/main.py
#LANCICHINETTI_FORTUNATO_SCRIPT=$PROJECT_DIR/Lancichinetti-Fortunato/main.py

#for DATASET_NAME in LFR-louvain LFR-mcl LFR
#for DATASET_NAME in LFR-preprocessed
for DATASET_NAME in LFR-louvain
do
    
    #for ALG in v8 boem saoem
    for ALG in lancichinetti-fortunato
    #for ALG in kirkley-newman
    #for ALG in v8-parallel
    #for ALG in boem
    do

        for N in 200 1000 5000 25000 125000
        #for N in 125000
        do
            #for MU in 01 02 03 04 05 06 07
            for MU in 04
            do
                OUTPUT_DIR_NAME="$ALG"."$DATASET_NAME".n"$N".mu"$MU"
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

                GRAPH_FILE=$DATA_DIR/$DATASET_NAME/n$N/LFR_n"$N"_mu"$MU"_gamma30_beta11.mtx
                INPUT_CLUSTERING_PREFIX=$DATA_DIR/$DATASET_NAME/n$N/LFR_n"$N"_mu"$MU"_gamma30_beta11
                OUTPUT_CLUSTERING_PREFIX=$OUTPUT_DIR/LFR_n"$N"_mu"$MU"_gamma30_beta11

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
                
                if [ "$ALG" == "kirkley-newman" ]; then
                    source /home/mth/.venv/bin/activate
                    python $KIRKLEY_NEWMAN_SCRIPT --graph-file "$GRAPH_FILE" \
                    --input-prefix "$INPUT_CLUSTERING_PREFIX" \
                    --k "$NUMBER_OF_INPUT_CLUSTERING" \
                    --output-prefix "$OUTPUT_CLUSTERING_PREFIX" \
                    --alg $ALG >> "$STDOUT_FILE"
                    deactivate
                elif [ "$ALG" == "lancichinetti-fortunato" ]; then
                    source /home/mth/.venv/bin/activate
                    python $LANCICHINETTI_FORTUNATO_SCRIPT --graph-file "$GRAPH_FILE" \
                    --input-prefix "$INPUT_CLUSTERING_PREFIX" \
                    --k "$NUMBER_OF_INPUT_CLUSTERING" \
                    --output-prefix "$OUTPUT_CLUSTERING_PREFIX" >> "$STDOUT_FILE"
                    deactivate
                elif [ "$ALG" == "v8-parallel" ]; then
                    # Default threshold for all runtime experiments
                    # Bring consensus of everything
                    for TRD in 1 2 4 8 16 32 64 128
                    do
                        export OMP_NUM_THREADS=$TRD
                        export OMP_PLACES=cores
                        T=0.99
                        $BIN --graph-file "$GRAPH_FILE" \
                        --input-prefix "$INPUT_CLUSTERING_PREFIX" \
                        --k "$NUMBER_OF_INPUT_CLUSTERING" \
                        --output-prefix "$OUTPUT_CLUSTERING_PREFIX" \
                        --pre-proc-threshold "$T" \
                        --alg $ALG >> "$STDOUT_FILE"."$TRD"
                    done
                else
                    $BIN --graph-file "$GRAPH_FILE" \
                    --input-prefix "$INPUT_CLUSTERING_PREFIX" \
                    --k "$NUMBER_OF_INPUT_CLUSTERING" \
                    --output-prefix "$OUTPUT_CLUSTERING_PREFIX" \
                    --alg $ALG >> "$STDOUT_FILE"
                fi
            done
        done
    done
done

