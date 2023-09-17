from typing import Union
import numpy as np
from torch import Tensor
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATv2Conv,SAGEConv,TransformerConv
from torch.nn import TransformerEncoderLayer



"""
有什么新想法可以在这个模型上面尝试进行修改
目前测试的想法有 双层gcn + transformerencoder
单层GraphSage+单层神经网络
双层GraphSage+双层神经网络
单层GraphSage（mean）+transformer+ 单层神经网络
单层GraphSage +双层mlp √ 目前best模型
准备进行尝试


近似度矩阵-》单层graph(max)----|
                            +---->双层mlp
空间矩阵-》单层graph(mean) ----| 


"""
similarity_matrix=np.load('similarity_matrix.npz')['matrix']
similarity_matrix=torch.tensor(similarity_matrix,dtype=torch.int64)
similarity_matrix = similarity_matrix.to('cuda:0')

class GCN(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, cached=True, aggr='mean'))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, cached=True))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, cached=True,aggr='mean'))

        self.convs2 = torch.nn.ModuleList()
        self.convs2.append(SAGEConv(in_channels, hidden_channels, cached=True, aggr='mean'))
        if self.batchnorm:
            self.bns2 = torch.nn.ModuleList()
            self.bns2.append(torch.nn.BatchNorm1d(hidden_channels))
            self.bns2.append(torch.nn.BatchNorm1d(64))
            self.bns2.append(torch.nn.BatchNorm1d(14))


        for _ in range(num_layers - 2):
            self.convs2.append(
                SAGEConv(hidden_channels, hidden_channels, cached=True))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs2.append(SAGEConv(hidden_channels, hidden_channels, cached=True, aggr='mean'))


        self.linear = torch.nn.ModuleList()
        self.linear.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.linear.append(torch.nn.Linear(hidden_channels, 64))
        self.linear.append(torch.nn.Linear(64, 14))
        self.linear.append(torch.nn.Linear(14, 2))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index: Union[Tensor]):
        x1=x
        x2=x
        '''
                for i, conv in enumerate(self.convs[:-1]):
            x1 = conv(x1, edge_index)
            if self.batchnorm:
                x1 = self.bns[i](x1)
            x1 = F.relu(x1)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        '''

        for i, conv in enumerate(self.convs2[:-1]):
            x2 = conv(x2, similarity_matrix)
            if self.batchnorm:
                x2 = self.bns[i](x2)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x=x2
        for i,lin in enumerate(self.linear[:-1]):
            x=lin(x)
            x=self.bns2[i](x)
            x=F.relu(x)
            x=F.dropout(x,p=self.dropout,training=self.training)
            #print(x.shape)
        x=self.linear[-1](x)
        #print(x.shape)
        return x.log_softmax(dim=-1)

class GATv2(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , num_heads
                 , batchnorm=True):
        super(GATv2, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=num_heads, concat=True))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels*num_heads))
        for i in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels*num_heads, hidden_channels, heads=num_heads, concat=True))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels*num_heads))
        self.convs.append(GATv2Conv(hidden_channels*num_heads
                          , out_channels
                          , heads=1
                          , concat=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index: Union[Tensor]):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)



class MLP(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)


class MLPLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPLinear, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x):
        return F.log_softmax(self.lin(x), dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index: Union[Tensor]):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)

class GraphTransformerGCN(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(GraphTransformerGCN,self).__init__()
        self.convs=torch.nn.ModuleList()
        self.bns=torch.nn.ModuleList()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers-2):
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(TransformerConv(hidden_channels,out_channels,heads=8))
        self.dropout=dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index: Union[Tensor]):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)
