# %% intial
import torch
import numpy
import time

from torch.utils.data import DataLoader

from utils.model_util import *
from utils.data_util import *
from utils.lib_util import *
from utils.train_util import *

t = time.localtime()
log_path = f'./log/{t.tm_year}-{t.tm_mon}-{t.tm_mday}-{t.tm_hour}-{t.tm_min}.log'
log = get_logger(log_path)

torch.set_printoptions(
    precision=2,
    threshold=1000,
    edgeitems=5,
    linewidth=1000,
    sci_mode=False)

if torch.cuda.is_available():
    device = 'cuda'
    log.info(f'device {device} is used.')
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        log.info('cudnn is actived.')
elif torch.backends.mps.is_available():
    device = 'mps'
    log.info(f'device {device} is used.')
else:
    device = 'cpu'
    log.info(f'device {device} is used.')


# %% 参数定义
args = get_args()
args.device = device

server_client = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
neighbor_server = [[1], [2], [0]]
all_client = [i for i in range(args.num_all_client)]
all_server = [i for i in range(args.num_all_server)]
num_server_client = args.num_all_client // args.num_all_server

message = f"\n\
{'alpha':^17}:{args.alpha:^7}\n\
{'T':^17}:{args.T:^7}\n\
{'algorithm':^17}:{args.algorithm:^7}\n\
{'num_server_commu':^17}:{args.num_server_commu:^7}\n\
{'num_client_commu':^17}:{args.num_client_commu:^7}\n\
{'num_client_train':^17}:{args.num_client_train:^7}\n\
{'num_public_train':^17}:{args.num_public_train:^7}\n\
{'batch_size':^17}:{args.batch_size:^7}\n\
{'dataset':^17}:{args.dataset:^7}\n\
{'model_select':^17}:{args.model_select:^7}\n\
{'num_all_client':^17}:{args.num_all_client:^7}\n\
{'num_all_server':^17}:{args.num_all_server:^7}\n\
{'num_client_data':^17}:{args.num_client_data:^7}\n\
{'num_public_data':^17}:{args.num_public_data:^7}\n\
{'proportion':^17}:{args.proportion:^7}"
log.info(message)
