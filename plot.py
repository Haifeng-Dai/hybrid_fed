import os

from matplotlib import plt

save_path = f'./data/dealed-data/{args.dataset}_algo_{args.algo}/alpha_{args.alpha}_T_{args.T}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'server_commu_{args.num_server_commu}_client_commu_{args.num_client_commu}_client_train_{args.num_client_train}_batch_size_{args.batch_size}_num_all_client_{args.num_all_client}_num_all_server_{args.num_all_server}_num_client_data_{args.num_client_data}_num_public_data_{args.num_public_data}_proportion_{args.proportion}_rank_{rank}.npz'
file os