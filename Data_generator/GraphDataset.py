import numpy
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


def read_data(folder):
    data_raw=np.load(folder)
    x = data_raw['x']
    y = data_raw['label'].reshape(-1, 1)
    edge_index = data_raw['edge_index']
    train_mask = data_raw['train_mask']
    valid_mask = data_raw['vaild_mask']
    test_mask = data_raw['test_mask']

    x = torch.tensor(x, dtype=torch.float).contiguous()
    y = torch.tensor(y, dtype=torch.int64)
    y=y.squeeze()
    edge_index = torch.tensor(edge_index, dtype=torch.int64).contiguous()
    train_mask = torch.tensor(train_mask, dtype=bool)
    valid_mask = torch.tensor(valid_mask, dtype=bool)
    test_mask = torch.tensor(test_mask, dtype=bool)

    #print(torch.nonzero(y[train_mask]).size(0))
    data = Data(x=x, edge_index=edge_index,  y=y)
    data.train_mask = train_mask
    data.valid_mask = valid_mask
    data.test_mask = test_mask

    return data

if __name__ == '__main__':
    data=read_data('data.npz')

