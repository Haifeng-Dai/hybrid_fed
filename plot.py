import os
import argparse

import matplotlib.pyplot as plt

from utils.lib_util import *

args = get_args()

save_path = f'./data/dealed-data/{args.dataset}_algo_{args.algorithm}/alpha_{args.alpha}_T_{args.T}/'

# file_name = f'server_commu_{args.num_server_commu}_client_commu_{args.num_client_commu}_client_train_{args.num_client_train}_batch_size_{args.batch_size}_num_all_client_{args.num_all_client}_num_all_server_{args.num_all_server}_num_client_data_{args.num_client_data}_num_public_data_{args.num_public_data}_proportion_{args.proportion}.npz'

file_path = save_path + (
    f'server_commu_{args.num_server_commu}'
    f'_client_commu_{args.num_client_commu}'
    f'_client_train_{args.num_client_train}'
    f'_batch_size_{args.batch_size}'
    f'_num_all_client_{args.num_all_client}'
    f'_num_all_server_{args.num_all_server}'
    f'_num_client_data_{args.num_client_data}'
    f'_num_public_data_{args.num_public_data}'
    f'_proportion_{args.proportion}.npz')

print(file_path)
