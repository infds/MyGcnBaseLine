import numpy
import pandas as pd
import numpy as np
import scipy.io
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sparse
import random
from sklearn.metrics import pairwise_distances
import torch

def read_data(path,shuffle) :
    data = pd.read_csv(path)
    data = data.drop(data.columns[0], axis=1)
    data = data.rename(columns={'a7': 'label'})
    if shuffle==True:
        data = data.sample(frac=1, random_state=42)
    return data

def OnlyOneHotEncode(data,columns_to_encode,sparse):
    encoder=OneHotEncoder(sparse=sparse)
    encode_data = encoder.fit_transform(data[columns_to_encode])
    encoded_df = pd.DataFrame(encode_data, columns=encoder.get_feature_names_out(columns_to_encode))
    data = pd.concat([data, encoded_df], axis=1)
    data = data.drop(data.columns[1], axis=1)
    data = data.drop(data.columns[2], axis=1)
    cols = data.columns.tolist()
    cols.append(cols.pop(5))
    data = data[cols]

    return data

def generate_adjacency_matrix(num_nodes,max_neighbors):
    adjacency_matrix=np.zeros((num_nodes,num_nodes))
    for node in range(num_nodes):
        num_neighbors=random.randint(1,max_neighbors)
        neighbors=random.sample(range(num_nodes),num_neighbors)
        adjacency_matrix[node,neighbors]=1
        adjacency_matrix[neighbors,node]=1

    return adjacency_matrix

def generate_Network(num_nodes,max_neighbors):
    adjancency_matrix = generate_adjacency_matrix(num_nodes, max_neighbors)
    Network = sparse.coo_matrix(adjancency_matrix)
    return Network

def generate_Attributes(data):
    atr_data = data.drop(data.columns[-1], axis=1)
    Attributes = atr_data.values
    Attributes = sparse.coo_matrix(Attributes)

    return Attributes

def generate_Label(data):
    label = data['label']
    Label = label.values

    return Label

def generate_Class(nums_sample):
    Class = np.zeros((nums_sample, 1))
    return Class

def save_as_siocoo(data,last_index,num_sample,mat_name,num_nodes):
    data=data[last_index:num_sample]
    Network=generate_Network(num_nodes,max_neighbors=3)
    Attributes=generate_Attributes(data)
    Label=generate_Label(data)
    Class=generate_Class(num_sample)
    data = {'Network': Network, 'Attributes': Attributes, 'Label': Label, 'Class': Class}
    scipy.io.savemat(mat_name, data)

def set_mask(num_nodes,num_masks):
    nodes=np.array(num_nodes)
    indices=np.random.choice(len(nodes),num_masks,replace=False)
    mask=np.zeros(len(nodes),dtype=bool)
    mask[indices]=True
    return mask

def split_mask(data):
    indices=np.arange(data.shape[0])
    np.random.shuffle(indices)
    print(indices)
    train_num=round(data.shape[0]*0.7)
    valid_num=round(data.shape[0]*0.2)
    test_num=data.shape[0]-train_num-valid_num

    train_indices=indices[:train_num]
    valid_indices=indices[train_num:train_num+valid_num]
    test_indices=indices[train_num+valid_num:]

    train_mask=np.zeros(data.shape[0],dtype=bool)
    train_mask[train_indices]=True
    vaild_mask = np.zeros(data.shape[0], dtype=bool)
    vaild_mask[valid_indices]=True
    test_mask=np.zeros(data.shape[0], dtype=bool)
    test_mask[test_indices]=True

    return train_mask,vaild_mask,test_mask


def get_label(data):
    label=data[:,-1]
    return label

def extract_edges(adjacency_matrix):
    num_nodes=adjacency_matrix.shape[0]
    edges=np.nonzero(adjacency_matrix)
    src_nodes=edges[0]
    dst_nodes=edges[1]

    all_edges=np.vstack((src_nodes,dst_nodes)).T
    all_edges=np.vstack((all_edges,np.flip(all_edges,axis=1)))

    return all_edges

def euclidean_similarity_matrix(features):
    dist_matrix=pairwise_distances(features,metric='euclidean')
    similarity_matrix=np.exp(-dist_matrix)
    np.fill_diagonal(similarity_matrix,0.0)
    return similarity_matrix

def k_nearest_neighbors(similarity_matrix,k):
    adjacency_matrix=np.zeros_like(similarity_matrix)
    num_nodes=similarity_matrix.shape[0]

    for i in range(num_nodes):
        k_nearest_indices=np.argsort(similarity_matrix[i])[-(k+1):-1]
        adjacency_matrix[i,k_nearest_indices]=1.0
        #adjacency_matrix[k_nearest_indices,i]=1.0

    return adjacency_matrix



if __name__=='__main__':

    """
    初始构造data数据
    data=read_data('Data/data.csv',shuffle=False)
    columns=['pipe-mat','先前失效']
    data=OnlyOneHotEncode(data,columns,sparse=False)
    edge_index=generate_adjacency_matrix(data.shape[0],3)
    label=get_label(data.values)
    train_mask,vaild_mask,test_mask=split_mask(data)
    edge=extract_edges(edge_index)
    edge=edge.transpose()
    x=data.values[:,:-1]
    np.savez('data.npz',x=x,label=label,train_mask=train_mask,vaild_mask=vaild_mask,test_mask=test_mask,edge_index=edge)
    """
    data=np.load('data.npz')
    similarity_matrix=euclidean_similarity_matrix(data['x'])
    adjacency_matrix=k_nearest_neighbors(similarity_matrix,3)
    all_edges=extract_edges(adjacency_matrix)
    all_edges=all_edges.transpose()
    print(all_edges.shape)
    np.savez('similarity_matrix.npz',matrix=all_edges)















