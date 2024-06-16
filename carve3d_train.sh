#!/bin/bash
NUM_NODES=$1

cd ddpo
pip install -e .
cd ..
# TODO configure accelerate

if [ "$NUM_NODES" -eq 1 ]; then
    accelerate launch carve3d_train.py --config config/dgx.py:carve3d_train
else
    accelerate launch --multi_gpu --num_machines $NUM_NODES --num_processes $(( $NUM_NODES * 8 )) --mixed_precision fp16 --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --machine_rank $RANK --same_network --rdzv_backend static --gpu_ids=all --rdzv_conf timeout=3600 carve3d_train.py --config config/dgx.py:carve3d_train
fi