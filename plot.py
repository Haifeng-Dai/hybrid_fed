import torch
import argparse
import numpy

import matplotlib.pyplot as plt

from utils.lib_util import *

args = get_args()

save_path = f'./data/dealed-data/{args.dataset}_algo_{args.algorithm}/alpha_{args.alpha}_T_{args.T}/'

# file_name = f'server_commu_{args.num_server_commu}_client_commu_{args.num_client_commu}_client_train_{args.num_client_train}_batch_size_{args.batch_size}_num_all_client_{args.num_all_client}_num_all_server_{args.num_all_server}_num_client_data_{args.num_client_data}_num_public_data_{args.num_public_data}_proportion_{args.proportion}.pt'

# file_path = save_path + (
#     f'server_commu_{args.num_server_commu}'
#     f'_client_commu_{args.num_client_commu}'
#     f'_client_train_{args.num_client_train}'
#     f'_batch_size_{args.batch_size}'
#     f'_num_all_client_{args.num_all_client}'
#     f'_num_all_server_{args.num_all_server}'
#     f'_num_client_data_{args.num_client_data}'
#     f'_num_public_data_{args.num_public_data}'
#     f'_proportion_{args.proportion}.pt')

file_path = './data/saved-data/mnist_algo_0/alpha_0.5_T_8/server_commu_10_client_commu_5_client_train_5_batch_size_200_num_all_client_9_num_all_server_3_num_client_data_1000_num_public_data_50_proportion_0.8.pt'
print(file_path)


# file_path = './data/dealed-data/mnist_algo_3/alpha_0.5_T_8/server_commu_2_client_commu_2_client_train_2_batch_size_10_num_all_client_4_num_all_server_2_num_client_data_100_num_public_data_50_proportion_0.8.pt'
data = torch.load(file_path)
# print(data.keys())

# a = data['server_model'][0][0].state_dict()
# b = data['server_model'][0][1].state_dict()
# for key in a.keys():
#     print(a[key] == b[key])

client_acc = data['client_acc']
b = numpy.array(client_acc)
for i in range(b.shape[0]):
    plt(client_acc[i, :])

plt.show()
