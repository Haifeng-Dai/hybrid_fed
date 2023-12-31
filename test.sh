#!/bin/bash

dataset='mnist'
alpha=0.5
T=8
num_all_client=9
num_all_server=3
batch_size=128
num_client_data=1200
num_server_commu=2
num_client_commu=2
num_client_train=2
num_public_train=2
model_select=1
algorithm=1
num_public_data=50
proportion=0.8

python data_obtain.py \
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

mpiexec -n ${num_all_client} --oversubscribe python test.py
