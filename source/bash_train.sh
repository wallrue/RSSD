#!/bin/bash
#NOTE: Modify BACKBONE_TEST, training_dict. Then, if any, dataset_dir, checkpoints_dir in train.py to choose the type of experiment and models to train
gpu_ids='0'
num_threads=2
batch_size=1

continue_train=false
epoch_count_start=1
epoch_pause_freq=10
niter=100
niter_decay=100

epoch_pause=$((epoch_count_start)) #backup the first epoch
epoch_checkpoint_load=$((epoch_count_start - 1))
save_epoch_freq=$(($((epoch_pause % $epoch_pause_freq))>0 ? epoch_pause : epoch_pause_freq))
loop_string=$(($(($((niter + niter_decay)) / $epoch_pause_freq)) - $((epoch_pause  / $epoch_pause_freq)))) #Traning models in parallel every "freq" epochs

loadSize=256
fineSize=256
validDataset_split=0.0
lr=0.0002

CMD="python train.py \
    --loadSize ${loadSize} \
    --fineSize ${fineSize} \
    --batch_size ${batch_size} \
    --gpu_ids ${gpu_ids} \
    --num_threads ${num_threads} \
    --save_epoch_freq ${save_epoch_freq} \
    --lr ${lr}\
    --niter ${niter} \
    --niter_decay ${niter_decay} \
    --epoch_pause ${epoch_pause}\
    --epoch_count ${epoch_count_start}\
    --epoch ${epoch_checkpoint_load}
    "
CMD1="--continue_train"

if $continue_train
then
  c="${CMD} ${CMD1}"
else
  c="${CMD}"
fi
echo $c
eval $c

for VARIABLE in $(seq 1 1 $loop_string)
do
    epoch_checkpoint_load=$((epoch_pause))
    epoch_count_start=$((epoch_pause + 1))
    epoch_pause=$(($(($((epoch_pause / $epoch_pause_freq)) + 1)) * $epoch_pause_freq))
    save_epoch_freq=$((epoch_pause_freq))
    CMD="python train.py \
        --loadSize ${loadSize} \
        --fineSize ${fineSize} \
        --batch_size ${batch_size} \
        --gpu_ids ${gpu_ids} \
        --num_threads ${num_threads} \
        --save_epoch_freq ${save_epoch_freq} \
        --lr ${lr}\
        --niter ${niter} \
        --niter_decay ${niter_decay} \
        --epoch_pause ${epoch_pause}\
        --epoch_count ${epoch_count_start}\
        --epoch ${epoch_checkpoint_load}\
        --continue_train\
        "
    c="${CMD}"
    echo $c
    eval $c
done
$SHELL #prevent bash window from closing



