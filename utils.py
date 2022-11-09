""" helper function

author axiumao
"""

import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset_true import My_Dataset

def get_network(args):
    """ return given network
    """

    if args.net == 'canet':
        from models.canet import canet
        net = canet()
    elif args.net == 'canet_wf_preprocess':
        from models.canet_wf_preprocess import canet
        net = canet()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_mydataloader(pathway, n_skip, data_id = 1, batch_size=16, num_workers=2, shuffle=True):
    Mydataset = My_Dataset(pathway, data_id, n_skip, transform=None)
    Data_loader = DataLoader(Mydataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    return Data_loader

def get_weighted_mydataloader(pathway, n_skip, data_id = 1, batch_size=16, num_workers=2, shuffle=True):
    Mydataset = My_Dataset(pathway, data_id, n_skip, transform=None)
    all_labels = [label for data, t_data, label in Mydataset]
    number = np.unique(all_labels, return_counts = True)[1]
    weight = 1./ torch.from_numpy(number).float()
    # print(weight)
    weight = torch.softmax(weight,dim=0)
    Data_loader = DataLoader(Mydataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    return Data_loader, weight, number
