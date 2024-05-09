#!/bin/bash -l

PROJECT_DIR=$HOME/Codes/graph-consensus-clustering
DATA_DIR=$PROJECT_DIR/test/data
RESULT_DIR=$PROJECT_DIR/test/experiment-results/benchmark-study

export OMP_NUM_THREADS=48
export OMP_PLACES=cores

BIN=$PROJECT_DIR/dist-distribution
AMI_SCRIPT=$PROJECT_DIR/ami.py

for DATASET_NAME in LFR-louvain LFR
#for DATASET_NAME in LFR-preprocessed
#for DATASET_NAME in LFR-louvain
do
    #for ALG in v8-parallel v8 boem saoem
    #for ALG in v8-parallel kirkley-newman
	#for ALG in lancichinetti-fortunato kirkley-newman v8-parallel
    #for ALG in kirkley-newman 
    #for ALG in v8-parallel 
    for ALG in boem
    do

        #for N in 200 1000 5000
        for N in 1000 5000
        #for N in 5000
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
                
                # Get number of input clusterings
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
                
                # Compute distance from consensus partition to ground truth partition
                for DIST in vi split-join rand nmi ami
                do
                    SOLN=0
                    while [ $SOLN -lt 100 ]
                    do
                        #echo Solution: $SOLN
                        CONSENSUS_FILE="$OUTPUT_CLUSTERING_PREFIX".soln-"$SOLN"
                        DISTANCE_DISTRIBUTION_FILE="$CONSENSUS_FILE".$DIST
                        #echo $DISTANCE_DISTRIBUTION_FILE
                        if [ -f $CONSENSUS_FILE ]; then
                            if [ "$DIST" == "ami" ]; then
                                source /home/mth/.venv/bin/activate
                                python $AMI_SCRIPT --input-file "$CONSENSUS_FILE" \
                                --gt-file "$GT_FILE" > "$DISTANCE_DISTRIBUTION_FILE"
                                deactivate
                            else
                                $BIN --consensus-file $CONSENSUS_FILE \
                                    --input-clustering-prefix $INPUT_CLUSTERING_PREFIX \
                                    --number-of-input-clustering $NUMBER_OF_INPUT_CLUSTERING \
                                    --ground-truth-file $GT_FILE \
                                    --output-file $DISTANCE_DISTRIBUTION_FILE \
                                    --distance-metric $DIST
                            fi
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
done
