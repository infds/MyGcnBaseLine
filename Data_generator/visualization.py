import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from models import SAGE,GCN,MLP
import argparse
import seaborn as sns
from sklearn.metrics import accuracy_score,precision_score,recall_score


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

def raw_data_viewTSNE(folder):
    data_raw = np.load(folder)
    feature = data_raw['x'][data_raw['train_mask']]
    labels = data_raw['label'].reshape(-1, 1)[data_raw['train_mask']]
    tsne=TSNE(n_components=2,random_state=42)
    embeded=tsne.fit_transform(feature)
    plt.figure(figsize=(10,10))
    colors = ["#2D7FB9" if label == 0 else "#FF8D29" for label in labels]
    plt.scatter(embeded[:,0],embeded[:,1],c=colors,s=10,alpha=0.7)
    #plt.colorbar()

    plt.title('t-SNE Visualization of Node Features')
    plt.xlabel('t-SNE Dimension')

    plt.show()


def raw_data_kde(folder):
    data_raw = np.load(folder)
    feature = data_raw['x'][data_raw['train_mask']]
    label = data_raw['label'].reshape(-1, 1)[data_raw['train_mask']]
    tsne = TSNE(n_components=2, random_state=42)
    embeded = tsne.fit_transform(feature)
    df = pd.DataFrame(embeded, columns=[f'feature_{i}' for i in range(2)])
    df['label'] = label
    plt.figure(figsize=(10, 8))
    sns.displot(data=df, x='feature_0', hue='label', kde=True,alpha=0.3,edgecolor='White',multiple='layer')
    #plt.axis('off')
    plt.show()



def trained_data_viewTSNE(data,feature):
    label=data.y[data.train_mask]
    labels=label.detach().cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    embeded = tsne.fit_transform(feature)
    plt.figure(figsize=(10, 10))
    colors=["#2D7FB9" if label == 0 else "#FF8D29"  for label in labels]
    plt.scatter(embeded[:, 0], embeded[:, 1], c=colors, s=10, alpha=1.0)
    # plt.colorbar()

    plt.title('t-SNE Visualization of Node Features')
    plt.xlabel('t-SNE Dimension')

    #plt.ylabel('t-SNE Dimension')

    plt.show()

def trained_data_KDE(data,feature):
    label = data.y[data.train_mask]
    label = label.detach().cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    embeded = tsne.fit_transform(feature)
    df=pd.DataFrame(embeded,columns=[f'feature_{i}' for i in range(2)])
    df['label']=label
    plt.figure(figsize=(10, 8))
    sns.displot(data=df,x='feature_0',hue='label',kde=True,alpha=0.3,edgecolor='White',multiple='layer')
    #plt.axis('off')
    #plt.ylabel('Density')
    plt.show()


val_metric = 'auc'

gcn_parameters = {'lr':0.01
              , 'num_layers':6
              , 'hidden_channels':128
              , 'dropout':0.5
              , 'batchnorm': True
              , 'l2':5e-7
             }

gat_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':256
              , 'dropout':0.5
              , 'batchnorm': True
              , 'l2':5e-7
             }

mlp_parameters = {'lr':0.001
              , 'num_layers':1
              , 'hidden_channels':64
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }
sage_parameters = {'lr':0.01
              , 'num_layers':3
              , 'hidden_channels':128
              , 'dropout':0.5
              , 'batchnorm': True
              , 'l2':5e-7
             }

trans_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':256
              , 'dropout':0.5
              , 'batchnorm': True
              , 'l2':5e-7
             }


features=[]

def hook(model,input,output):
    features.append(input)
    return None

similarity_matrix=np.load('similarity_matrix.npz')['matrix']
similarity_matrix=torch.tensor(similarity_matrix,dtype=torch.int64)
similarity_matrix = similarity_matrix.to('cuda:0')


if __name__ =='__main__':
    #raw_data_viewTSNE('data.npz')

    parser = argparse.ArgumentParser(description='gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='data.npz')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--fold', type=int, default=0)

    args = parser.parse_args()
    print(args)

    no_conv = False
    if args.model in ['mlp']: no_conv = True

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    data = read_data('data.npz')
    data = data.to(device)
    split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}
    #print(torch.nonzero(data.y[data.train_mask]))
    train_idx = split_idx['train'].to(device)

    if args.model == 'gcn':
        para_dict = gcn_parameters
        model_para = gcn_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = GCN(in_channels=data.x.size(-1), out_channels=2, **model_para).to(device)




    print(f'Model {args.model} initialized')

    model.load_state_dict(torch.load('best_model.pth'))

    for (name,module) in model.named_modules():
        print(name)
    model.linear[2].register_forward_hook(hook)
    y_pred=model(data.x,similarity_matrix)
    torch.set_printoptions(precision=8, sci_mode=False)
    y_pred = y_pred.detach().cpu().numpy()
    np.savetxt('model.csv', y_pred, fmt='%.2f', delimiter=',')
    mask=data['test_mask']
    mask=mask.detach().cpu().numpy()
    y_pred=y_pred[mask].argmax(axis=-1)

    #y_pred = y_pred[data['test_mask']]
    label=data.y[data["test_mask"]]
    label = label.detach().cpu().numpy()
    test_precision01=precision_score(y_pred,label)
    test_precision02 = accuracy_score(y_pred, label)
    print(test_precision01,test_precision02)
    """  
    print(features[0][0].shape)
    features=features[0][0].detach().cpu().numpy()
    features=features[split_idx['train'].detach().cpu().numpy()]#只进行测试集样本的ksne分布

    #raw_data_viewTSNE('data.npz')
    #raw_data_kde('data.npz')
    trained_data_viewTSNE(data,features)
    trained_data_KDE(data,features)

    np.savez("../HLWproject/dl_result.npz",y_pred=y_pred,label=label)
    fpr_RFC01, tpr_RFC01, thresholds = roc_curve(label, y_pred[:,1])
    roc_auc_RFC01 = auc(fpr_RFC01, tpr_RFC01)
    print(roc_auc_RFC01)
    """







