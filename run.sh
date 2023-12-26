#!/bin/bash

alpha=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
T=(1 2 3 4 5 6 7 8 9 10)
batch_size=200
algorithm=(1 2 3 4)

dataset=(mnist cifar10 cifar100)
model=(1 2 3 4 5 6 7)

for alpha_ in ${alpha[@]}; do
    for T_ in ${T[@]}; do
        for dataset_ in ${dataset[@]}; do
            for algorithm_ in ${algorithm[@]}; do
                python test.py \
                    --dataset ${dataset_} \
                    --alpha ${alpha_} \
                    --T ${T_} \
                    --num_all_client 9 \
                    --num_all_server 3 \
                    --batch_size ${batch_size} \
                    --num_client_data 1000 \
                    --num_server_commu 200 \
                    --num_client_commu 10 \
                    --num_client_train 10 \
                    --num_public_train 10 \
                    --model_select 1 \
                    --algorithm ${algorithm_} \
                    --num_public_data 50 \
                    --proportion 0.8
            done
        done
    done
done
