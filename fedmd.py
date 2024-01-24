from utils import *

train_validate_set, test_set, c, h, w = get_dataset()
client_set = non_iid(dataset=train_validate_set,
                     alpha=0.1,
                     num_client=9,
                     num_client_data=4800)

