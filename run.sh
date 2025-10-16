#!/bin/bash

# source activate your_env_name
# sudo iptables -A INPUT -p tcp --dport 5920 -j ACCEPT

export OMP_NUM_THREADS=32
export NCCL_P2P_DISABLE=1 
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 get_label.py --config ./config/MGN_lid2d_stationary_ts_train.json

torchrun --nproc_per_node=1 main.py --config ./config/MGN_lid2d_stationary_ts_train.json
