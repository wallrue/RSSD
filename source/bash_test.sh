#!/bin/bash
#NOTE: Modify BACKBONE_TEST, training_dict. Then, if any, dataset_dir, checkpoints_dir in test.py to choose models to test
gpu_ids='0'
num_threads=2
batch_size=4
loop_string=200

loadSize=256
fineSize=256

for VARIABLE in latest #$(seq 10 10 $loop_string)
do
    epoch_checkpoint_load=$((VARIABLE))
    CMD="python test.py \
        --loadSize ${loadSize} \
        --fineSize ${fineSize} \
        --batch_size ${batch_size} \
        --gpu_ids ${gpu_ids} \
        --epoch ${epoch_checkpoint_load}\
        "
    c="${CMD}"
    echo $c
    eval $c
done
$SHELL #prevent bash window from closing
