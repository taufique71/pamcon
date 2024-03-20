#!/bin/bash -l

PROJECT_DIR=$HOME/Codes/graph-consensus-clustering
DATA_DIR=$PROJECT_DIR/test/data
RESULT_DIR=$PROJECT_DIR/test/experiment-results

export OMP_NUM_THREADS=8
export OMP_PLACES=cores

BIN=$PROJECT_DIR/consensus
KIRKLEY_NEWMAN_SCRIPT=$PROJECT_DIR/Kirkley-Newman/main.py

#for DATASET_NAME in LFR-louvain LFR-mcl LFR
#for DATASET_NAME in LFR-preprocessed
for DATASET_NAME in LFR
do
    
    #for ALG in v8 boem saoem
    #for ALG in v8-parallel
    for ALG in v8-parallel kirkley-newman
    do

        for N in 1000 5000
        #for N in 5000
        do
            for MU in 01 02 03 04 05 06 07
            #for MU in 06
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
                else
                    #valgrind --leak-check=full \
                        #--show-leak-kinds=all \
                        #--track-origins=yes \
                        #--verbose \
                        #--log-file=valgrind-out.txt \
                    $BIN --graph-file "$GRAPH_FILE" \
                    --input-prefix "$INPUT_CLUSTERING_PREFIX" \
                    --k "$NUMBER_OF_INPUT_CLUSTERING" \
                    --output-prefix "$OUTPUT_CLUSTERING_PREFIX" \
                    --pre-proc-threshold 0.45 \
                    --alg $ALG >> "$STDOUT_FILE"
                fi
            done
        done
    done
done

