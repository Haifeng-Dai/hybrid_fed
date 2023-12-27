#!/bin/bash
dataset='mnist'
alpha=0.5
T=8
num_all_client=9
num_all_server=3
batch_size=200
num_client_data=1000
num_server_commu=20
num_client_commu=10
num_client_train=10
num_public_train=10
model_select=1
algorithm=3
num_public_data=50
proportion=0.8
server_client='[[0, 1, 2], [3, 4, 5], [6, 7, 8]]'
neighbor_server='[[1], [2], [0]]'

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

# python test-1.py \
#     --dataset ${dataset} \
#     --alpha ${alpha} \
#     --T ${T} \
#     --num_all_client ${num_all_client} \
#     --num_all_server ${num_all_server} \
#     --batch_size ${batch_size} \
#     --num_client_data ${num_client_data} \
#     --num_server_commu ${num_server_commu} \
#     --num_client_commu ${num_client_commu} \
#     --num_client_train ${num_client_train} \
#     --num_public_train ${num_public_train} \
#     --model_select ${model_select} \
#     --algorithm ${algorithm} \
#     --num_public_data ${num_public_data} \
#     --proportion ${proportion}
