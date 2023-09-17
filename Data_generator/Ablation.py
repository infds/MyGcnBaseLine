import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
import pandas as pd
import evaluator
from torch_geometric.data import Data
import numpy as np

from models import GCN,GATv2,MLP,SAGE,GraphTransformerGCN
from evaluator import Evaluator
from logger import Logger

from target import prepare_folder

from GraphDataset import read_data

import matplotlib.pyplot as plt

eval_metric = 'auc'

gcn_parameters = {'lr':0.01
              , 'num_layers':6
              , 'hidden_channels':128
              , 'dropout':0.5
              , 'batchnorm': True
              , 'l2':5e-7
             }

gat_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.5
              , 'batchnorm': True
              , 'l2':5e-7
             }

mlp_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.4
              , 'batchnorm': True
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


"""
使用相似性矩阵
"""
similarity_matrix=np.load('similarity_matrix.npz')['matrix']
similarity_matrix=torch.tensor(similarity_matrix,dtype=torch.int64)
similarity_matrix = similarity_matrix.to('cuda:0')




def train(model, data, train_idx, optimizer, no_conv=False):
    # data.y is labels of shape (N, )
    model.train()

    optimizer.zero_grad()
    if no_conv:
        out = model(data.x[train_idx])
    else:
        #out = model(data.x, data.edge_index)[train_idx]
        out = model(data.x, similarity_matrix)[train_idx]

    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def train01(model, data, train_idx, optimizer, no_conv=False):
    # data.y is labels of shape (N, )
    model.train()

    optimizer.zero_grad()
    if no_conv:
        out = model(data.x[train_idx])
    else:
        out = model(data.x, data.edge_index)[train_idx]
        #out = model(data.x, similarity_matrix)[train_idx]
    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()
@torch.no_grad()
def test(model, data, split_idx, evaluator, no_conv=False):
    # data.y is labels of shape (N, )
    model.eval()

    if no_conv:
        out = model(data.x)
    else:
        #out = model(data.x, data.edge_index)
        out = model(data.x, similarity_matrix)

    y_pred = out.exp()  # (N,num_classes)

    losses, eval_results = dict(), dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()

        eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]

    return eval_results, losses, y_pred


@torch.no_grad()
def test01(model, data, split_idx, evaluator, no_conv=False):
    # data.y is labels of shape (N, )
    model.eval()

    if no_conv:
        out = model(data.x)
    else:
        out = model(data.x, data.edge_index)
        #out = model(data.x, similarity_matrix)

    y_pred = out.exp()  # (N,num_classes)

    losses, eval_results = dict(), dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()

        eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]

    return eval_results, losses, y_pred

def main():
    parser = argparse.ArgumentParser(description='gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='data.npz')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)

    args = parser.parse_args()
    print(args)

    no_conv = False
    if args.model in ['mlp']: no_conv = True

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    data=read_data('data.npz')
    data=data.to(device)
    split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}
    print(torch.nonzero(data.y[data.train_mask]))
    train_idx = split_idx['train'].to(device)

    result_dir = prepare_folder(args.dataset, args.model)
    print('result_dir:', result_dir)

    if args.model == 'gcn':
        para_dict = gcn_parameters
        model_para = gcn_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = GCN(in_channels=data.x.size(-1),out_channels=2, **model_para).to(device)

    if args.model == 'gat':
        para_dict = gcn_parameters
        model_para = gcn_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = GATv2(in_channels=data.x.size(-1),out_channels=2,num_heads=8, **model_para).to(device)

    if args.model == 'mlp':
        para_dict = mlp_parameters
        model_para = mlp_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = MLP(in_channels = data.x.size(-1), out_channels = 2, **model_para).to(device)

    if args.model == 'sage':
        para_dict = sage_parameters
        model_para = sage_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = SAGE(in_channels=data.x.size(-1), out_channels=2, **model_para).to(device)

    if args.model == 'trans':
        para_dict = trans_parameters
        model_para = trans_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = GraphTransformerGCN(in_channels=data.x.size(-1), out_channels=2, **model_para).to(device)

    print(f'Model {args.model} initialized')


    evaluator = Evaluator(eval_metric)
    logger = Logger(args.runs, args)
    list_acc=[]

    for run in range(args.runs):
        import gc
        gc.collect()
        print(sum(p.numel() for p in model.parameters()))

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
        best_valid = 0
        best_test=0
        min_valid_loss = 1e8
        best_out = None

        for epoch in range(1, args.epochs + 1):
            loss = train(model, data, train_idx, optimizer, no_conv)
            eval_results, losses, out = test(model, data, split_idx, evaluator, no_conv)
            train_eval, valid_eval, test_eval = eval_results['train'], eval_results['valid'], eval_results['test']
            train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

            #                 if valid_eval > best_valid:
            #                     best_valid = valid_result
            #                     best_out = out.cpu().exp()
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                best_out = out.cpu()

            if test_eval>best_test:
                best_test=test_eval
                torch.save(model.state_dict(),'best_model.pth')


            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_eval:.3f}%, '
                      f'Valid: {100 * valid_eval:.3f}% '
                      f'Test: {100 * test_eval:.3f}%')
            list_acc.append(100 * test_eval)
            logger.add_result(run, [train_eval, valid_eval, test_eval])

        logger.print_statistics(run)

    para_dict = sage_parameters
    sage_para = sage_parameters.copy()
    sage_para.pop('lr')
    sage_para.pop('l2')
    model01 = SAGE(in_channels=data.x.size(-1), out_channels=2, **sage_para).to(device)

    list_acc01 = []
    for run in range(args.runs):
        import gc
        gc.collect()
        print(sum(p.numel() for p in model01.parameters()))

        model01.reset_parameters()
        optimizer = torch.optim.Adam(model01.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
        best_valid = 0
        best_test = 0
        min_valid_loss = 1e8
        best_out = None

        for epoch in range(1, args.epochs + 1):
            loss = train01(model01, data, train_idx, optimizer, no_conv)
            eval_results, losses, out = test01(model01, data, split_idx, evaluator, no_conv)
            train_eval, valid_eval, test_eval = eval_results['train'], eval_results['valid'], eval_results['test']
            train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

            #                 if valid_eval > best_valid:
            #                     best_valid = valid_result
            #                     best_out = out.cpu().exp()
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                best_out = out.cpu()

            if test_eval > best_test:
                best_test = test_eval
                torch.save(model.state_dict(), 'best_model.pth')

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_eval:.3f}%, '
                      f'Valid: {100 * valid_eval:.3f}% '
                      f'Test: {100 * test_eval:.3f}%')
            list_acc01.append(100 * test_eval)
            logger.add_result(run, [train_eval, valid_eval, test_eval])

        logger.print_statistics(run)

    para_dict = gat_parameters
    gat_para = gat_parameters.copy()
    gat_para.pop('lr')
    gat_para.pop('l2')
    model02 = SAGE(in_channels=data.x.size(-1), out_channels=2, **gat_para).to(device)

    list_acc02 = []
    for run in range(args.runs):
        import gc
        gc.collect()
        print(sum(p.numel() for p in model01.parameters()))

        model01.reset_parameters()
        optimizer = torch.optim.Adam(model02.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
        best_valid = 0
        best_test = 0
        min_valid_loss = 1e8
        best_out = None

        for epoch in range(1, args.epochs + 1):
            loss = train(model02, data, train_idx, optimizer, no_conv)
            eval_results, losses, out = test(model02, data, split_idx, evaluator, no_conv)
            train_eval, valid_eval, test_eval = eval_results['train'], eval_results['valid'], eval_results['test']
            train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

            #                 if valid_eval > best_valid:
            #                     best_valid = valid_result
            #                     best_out = out.cpu().exp()
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                best_out = out.cpu()

            if test_eval > best_test:
                best_test = test_eval
                torch.save(model.state_dict(), 'best_model.pth')

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_eval:.3f}%, '
                      f'Valid: {100 * valid_eval:.3f}% '
                      f'Test: {100 * test_eval:.3f}%')
            list_acc02.append(100 * test_eval)
            logger.add_result(run, [train_eval, valid_eval, test_eval])

        logger.print_statistics(run)


    iterations=list(range(1,501))
    plt.plot(iterations,list_acc,linestyle="-",color='#619ac3',label='trick A,B,C')
    plt.plot(iterations, list_acc01, linestyle="-", color='#ed6d46', label='trick A')
    plt.plot(iterations, list_acc02, linestyle="-", color='#FBDD85', label='trick A,B')

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')

    plt.grid(True)
    plt.legend()
    plt.show()




if __name__ == '__main__':
    main()