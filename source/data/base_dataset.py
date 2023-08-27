###############################################################################
# This file contains the base dataset class which will be inherited by 
# other child dataset classes
###############################################################################

import torch.utils.data as data

class BaseDataset(data.Dataset):
    def name(self):
        return 'BaseDataset'
    
    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        return 0
