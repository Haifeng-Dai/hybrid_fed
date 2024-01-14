#!/bin/bash

dataset=mnist # (mnist cifar10 cifar100)
alpha=0.1       # (0.0 0.2 0.4 0.6 0.8 1.0)
T=6       # (2 4 6 8)
num_all_client=9
num_all_server=3
batch_size=160
num_client_data=1200
num_server_commu=50
num_client_commu=5
num_client_train=5
num_public_train=5
model_select=1 # (1 2 3)
algorithm=1        # (0 1 2 3 4)
num_public_data=50
proportion=0.8

# for alpha_ in ${alpha[@]}; do
#     for T_ in ${T[@]}; do
# for dataset_ in ${dataset[@]}; do
#     for model_select_ in ${model_select[@]}; do
python hybrid_fed.py \
    --dataset ${dataset} \
    --alpha ${alpha} \
    --T ${T} \
    --num_all_client ${num_all_client} \
    --num_all_server ${num_all_server} \
    --batch_size ${batch_size} \
    --num_client_data ${num_client_data} \
    --num_server_commu ${num_server_commu} \
    --num_client_commu ${num_client_commu} \
    --num_client_train ${num_client_train} \
    --num_public_train ${num_public_train} \
    --model_select ${model_select} \
    --algorithm ${algorithm} \
    --num_public_data ${num_public_data} \
    --proportion ${proportion}

        # python plot.py \
        #     --dataset ${dataset_} \
        #     --alpha ${alpha_} \
        #     --T ${T_} \
        #     --num_all_client 9 \
        #     --num_all_server 3 \
        #     --batch_size ${batch_size} \
        #     --num_client_data 1000 \
        #     --num_server_commu 200 \
        #     --num_client_commu 10 \
        #     --num_client_train 10 \
        #     --num_public_train 10 \
        #     --model_select 1 \
        #     --algorithm ${algorithm_} \
        #     --num_public_data 50 \
        #     --proportion 0.8
#     done
# done
#     done
# done
