#!/bin/bash -l

PROJECT_DIR=$HOME/Codes/graph-consensus-clustering
DATA_DIR=$PROJECT_DIR/test/data
RESULT_DIR=$PROJECT_DIR/test/experiment-results/benchmark-study

export OMP_NUM_THREADS=48
export OMP_PLACES=cores

BIN=$PROJECT_DIR/dist-distribution
AMI_SCRIPT=$PROJECT_DIR/ami.py

#for DATASET_NAME in LFR-louvain LFR-mcl LFR
#for DATASET_NAME in LFR-preprocessed
#for DATASET_NAME in LFR-louvain LFR
for DATASET_NAME in LFR-louvain-1 LFR-louvain-2 LFR-louvain-3 LFR-louvain-4 LFR-louvain-5 LFR-louvain-6 LFR-louvain-7 LFR-louvain-8 LFR-louvain-9 LFR-louvain-10
#for DATASET_NAME in LFR-1 LFR-2 LFR-3 LFR-4 LFR-5 LFR-6 LFR-7 LFR-8 LFR-9 LFR-10
do
    #for N in 200 1000 5000
    for N in 5000
    do
		for MU in 01 02 03 04 05 06 07
		#for MU in 01
        do
            FILE_PREFIX=LFR_n"$N"_mu"$MU"_gamma30_beta11
            INPUT_CLUSTERING_PREFIX=$DATA_DIR/$DATASET_NAME/n$N/$FILE_PREFIX
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

			#for DIST in nmi
			#for DIST in ami
            for DIST in vi split-join rand
			do
                ## Compute distance from ground truth to all input partitions
                #K=0
                #while [ $K -lt $NUMBER_OF_INPUT_CLUSTERING ]
                #do
                    #IP_FILE="$INPUT_CLUSTERING_PREFIX"."$K"
                    #DISTANCE_FILE="$IP_FILE".$DIST.gt
                    #if [ -f $IP_FILE ]; then
                        #if [ "$DIST" == "ami" ]; then
                            #source /home/mth/.venv/bin/activate
                            #python $AMI_SCRIPT --file-1 "$IP_FILE" \
                            #--file-2 "$GT_FILE" > "$DISTANCE_FILE"
                            #deactivate
                        #else
                            #$BIN --file-1 $IP_FILE \
                                #--file-2 $GT_FILE \
                                #--output-file $DISTANCE_FILE \
                                #--distance-metric $DIST
                        #fi
                    #else
                        #break;
                    #fi
                    #echo ---
                    #K=$(($K+1))
                #done

                ## Compute distance from ground truth to all consensus partitions
                for ALG in v8-parallel kirkley-newman lancichinetti-fortunato
                do
                    OUTPUT_DIR_NAME="$ALG"."$DATASET_NAME".n"$N".mu"$MU"
                    OUTPUT_DIR="$RESULT_DIR"/"$OUTPUT_DIR_NAME"
                    OUTPUT_CLUSTERING_PREFIX=$OUTPUT_DIR/$FILE_PREFIX

                    SOLN=0
                    while [ $SOLN -lt 100 ]
                    do
                        CONSENSUS_FILE="$OUTPUT_CLUSTERING_PREFIX".soln-"$SOLN"
                        DISTANCE_FILE="$CONSENSUS_FILE".$DIST.gt
                        echo $CONSENSUS_FILE
                        if [ -f $CONSENSUS_FILE ]; then
                            echo $CONSENSUS_FILE
                            if [ "$DIST" == "ami" ]; then
                                source /home/mth/.venv/bin/activate
                                python $AMI_SCRIPT --file-1 "$CONSENSUS_FILE" \
                                --file-2 "$GT_FILE" > "$DISTANCE_FILE"
                                deactivate
                            else
                                $BIN --file-1 $CONSENSUS_FILE \
                                    --file-2 $GT_FILE \
                                    --output-file $DISTANCE_FILE \
                                    --distance-metric $DIST
                            fi
                        else
                            break;
                        fi
                        echo ---
                        SOLN=$(($SOLN+1))
                    done
                done

                ## Compute distance from each consensus partition to all input partitions
                #for ALG in v8-parallel kirkley-newman lancichinetti-fortunato boem
                for ALG in boem
                do
                    OUTPUT_DIR_NAME="$ALG"."$DATASET_NAME".n"$N".mu"$MU"
                    OUTPUT_DIR="$RESULT_DIR"/"$OUTPUT_DIR_NAME"
                    OUTPUT_CLUSTERING_PREFIX=$OUTPUT_DIR/$FILE_PREFIX

                    SOLN=0
                    while [ $SOLN -lt 100 ]
                    do
                        CONSENSUS_FILE="$OUTPUT_CLUSTERING_PREFIX".soln-"$SOLN"
                        echo $CONSENSUS_FILE
                        if [ -f $CONSENSUS_FILE ]; then
                            K=0
                            while [ $K -lt $NUMBER_OF_INPUT_CLUSTERING ]
                            do
                                IP_FILE="$INPUT_CLUSTERING_PREFIX"."$K"
                                DISTANCE_FILE="$CONSENSUS_FILE".$DIST.$K
                                if [ -f $IP_FILE ]; then
                                    if [ "$DIST" == "ami" ]; then
                                        source /home/mth/.venv/bin/activate
                                        python $AMI_SCRIPT --file-1 "$IP_FILE" \
                                        --file-2 "$CONSENSUS_FILE" > "$DISTANCE_FILE"
                                        deactivate
                                    else
                                        $BIN --file-1 $IP_FILE \
                                            --file-2 $CONSENSUS_FILE \
                                            --output-file $DISTANCE_FILE \
                                            --distance-metric $DIST
                                    fi
                                else
                                    break;
                                fi
                                echo ---
                                K=$(($K+1))
                            done
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
