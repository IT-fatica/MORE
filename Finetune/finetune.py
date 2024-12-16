import warnings; warnings.filterwarnings('ignore') ##

import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import shutil

from data.splitters import scaffold_split, random_split, random_scaffold_split ##
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.base_model import GNN_graphpred ##
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error ## for regression
from tensorboardX import SummaryWriter

from datetime import datetime ##

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer):
    model.train()

    total_loss = 0.0 ## addition for calcuate the total loss
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()
        
        total_loss += loss.item()
    
    ## calcuate the total loss and record by tensorboard
    avg_train_loss = total_loss / len(loader)
    
    return avg_train_loss ## addition: return avg_train_loss

def train_reg(args, model, device, loader, optimizer):
    # code from HiMol's finetune.py
    model.train()
    
    total_loss = 0.0 ## addition for calcuate the total loss
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        
        if args.dataset in ['qm7', 'qm8', 'qm9']:
            loss = torch.sum(torch.abs(pred-y)) / y.size(0)
        elif args.dataset in ['esol', 'freesolv', 'lipophilicity']:
            loss = torch.sum((pred-y)**2) / y.size(0)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    ## calcuate the total loss and record by tensorboard
    avg_train_loss = total_loss / len(loader)
    
    return avg_train_loss ## addition: return avg_train_loss

def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]

def eval_reg(args, model, device, loader):
    # code from HiMol's finetune.py
    model.eval()
    y_true = []
    y_scores = []
    
    for step, batch in enumerate(tqdm(loader, desc='Iteration')):
        batch = batch.to(device)
        
        with torch.no_grad():
            pred = model(batch)
            
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)
        
    y_true = torch.cat(y_true, dim=0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy().flatten()
    
    mse = mean_squared_error(y_true, y_scores)
    mae = mean_absolute_error(y_true, y_scores)
    rmse = np.sqrt(mean_squared_error(y_true,y_scores))

    return mse, mae, rmse

def define_parser(parser: argparse.ArgumentParser):
    ## original
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'bace', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=-1, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--num_workers', type=int, default = 0, help='number of workers for dataset loading')
    
    ## modify
    parser.add_argument('--eval_train', action='store_true', help='evaluating training or not')
    parser.add_argument('--ft_type', type=str, default='full', help='Type of fine-tune (full, freeze, no_pretrain)')
    
    ## addition
    parser.add_argument('--run_cv', type=int, default=1, help='Number of iterations')
    parser.add_argument('--seed_list_in', type=str, default='', help="seed list")
    parser.add_argument('--seed_list', type=list, default=[], help="seed list")

def exec_main(args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    ## seed
    if args.runseed != -1: ## random seed if runseed is -1, or the specified seed if it is a positive number above 0.
        torch.manual_seed(args.runseed)
        np.random.seed(args.runseed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.runseed)

    ## task_type deffine
    if args.dataset in ['tox21', 'hiv', 'pcba', 'muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox', 'mutag']:
        task_type = 'cls'
    else:
        task_type = 'reg'
    
    #Bunch of classification tasks
    
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == 'esol':
        num_tasks = 1
    elif args.dataset == 'freesolv':
        num_tasks = 1
    elif args.dataset == 'lipophilicity':
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    print('### Setup Dataset ###')
    data_root = "dataset/"
    from data.loader import MoleculeDataset ##
    dataset = MoleculeDataset(data_root + args.dataset, dataset=args.dataset)

    print('[dataset]') ##
    print(dataset)
    
    if args.split == "scaffold":
        print("scaffold spllit ...")
        smiles_list = pd.read_csv(data_root + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
    elif args.split == "random":
        print("random split ...")
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    elif args.split == "random_scaffold":
        print("random scaffold split ...")
        smiles_list = pd.read_csv(data_root + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    else:
        raise ValueError("Invalid split option.")

    print('[train_dataset example]') ##
    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model 
    print('### Setup Model ###')
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if args.input_model_file != "" and args.ft_type != 'no_pretrain':
        print('Load Pretrained parameter ...')

        if "SimSGT" in args.input_model_file:
            model_file = os.path.join('pretrain', args.input_model_file)
            msg = model.gnn.load_state_dict(torch.load(model_file), map_location='cpu')
            print(msg)
    
        else:        
            model.from_pretrained(args.input_model_file, device)
        
    
    model.to(device)
    print('[model]') ##
    print(model) ##

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    if args.ft_type in ['full', 'no_pretrain', 'freeze']: ## original
        if args.ft_type != 'freeze':
            model_param_group.append({"params": model.gnn.parameters()})
        
        if args.graph_pooling == "attention":
            model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
        
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    
    else:
        raise ValueError("Invalid fine-tune type option.")
    
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)
    
    ## calculate the number of learnable parameter
    total_param = 0
    for group in model_param_group:
        total_param += sum(param.numel() for param in group['params'] if param.requires_grad)
    print(f'Number of learnable parameter: {total_param:,}')

    if task_type == 'cls':
        train_acc_list = []
        val_acc_list = []
        test_acc_list = []

        best_test_auc = [0, 0, 0, 0] # order: [train ROC-AUC, valid ROC-AUC, test ROC-AUC, best ROC-AUC epoch]
        for epoch in range(1, args.epochs+1):
            print("==== epoch " + str(epoch))
            
            train_loss = train(args, model, device, train_loader, optimizer) ## modify: get train_loss
            print("train LOSS: %f" % (train_loss)) ## addition: print train loss
            
            print("==== Evaluation")
            if args.eval_train:
                train_acc = eval(args, model, device, train_loader)
            else:
                print("omit the training accuracy computation")
                train_acc = 0
                
            val_acc = eval(args, model, device, val_loader)
            test_acc = eval(args, model, device, test_loader)

            print("train AUC: %f val AUC: %f test AUC: %f" % (train_acc, val_acc, test_acc)) ## modify: style
            
            if best_test_auc[1] < val_acc: # evaluate based on validation sets
                best_test_auc = [train_acc, val_acc, test_acc, epoch]

            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            train_acc_list.append(train_acc)
            print()
            
    elif task_type == 'reg':
        train_mse_list, train_mae_list, train_rmse_list = [], [], []
        val_mse_list, val_mae_list, val_rmse_list = [], [], []
        test_mse_list, test_mae_list, test_rmse_list = [], [], []
        
        best_test_mse = [1000, 1000, 1000, 0] # order: [train ROC-AUC, valid ROC-AUC, test ROC-AUC, best ROC-AUC epoch]
        for epoch in range(1, args.epochs+1):
            print("==== epoch " + str(epoch))
            
            train_loss = train_reg(args, model, device, train_loader, optimizer) ## modify: get train_loss
            print("train LOSS: %f" % (train_loss)) ## addition: print train loss
            
            print("==== Evaluation")
            if args.eval_train:
                train_mse, train_mae, train_rmse = eval_reg(args, model, device, train_loader)
            else:
                print("omit the training accuracy computation")
                train_mse, train_mae, train_rmse = 0, 0, 0
                
            val_mse, val_mae, val_rmse = eval_reg(args, model, device, val_loader)
            test_mse, test_mae, test_rmse = eval_reg(args, model, device, test_loader)

            if args.dataset in ['esol', 'freesolv', 'lipophilicity']:
                print("train RMSE: %.6f val RMSE: %.6f test RMSE: %.6f" % (train_rmse, val_rmse, test_rmse))

                if best_test_mse[1] > val_rmse: # evaluate based on validation sets
                    best_test_mse = [train_rmse, val_rmse, test_rmse, epoch]

            elif args.dataset in ['qm7', 'qm8', 'qm9']:
                print("train MAE: %.6f val MAE: %.6f test MAE: %.6f" % (train_mae, val_mae, test_mae))
                
                if best_test_mse[1] > val_mae: # evaluate based on validation sets
                    best_test_mse = [train_mae, val_mae, test_mae, epoch]

            train_mse_list.append(train_mse); train_mae_list.append(train_mae); train_rmse_list.append(train_rmse)
            val_mse_list.append(val_mse); val_mae_list.append(val_mae); val_rmse_list.append(val_rmse)
            test_mse_list.append(test_mse); test_mae_list.append(test_mae); test_rmse_list.append(test_rmse)

            print()
        
    if task_type == 'cls':
        best_test_auc.append('cls')
        return best_test_auc
    
    elif task_type == 'reg':
        best_test_mse.append('reg')
        return best_test_mse

def modify_seed_list(args):
    # str -> int
    args.seed_list = [int(i) for i in args.seed_list_in.split(' ')]
    args.seed_list.sort()
    
def main():
    # Training settings
    parser = argparse.ArgumentParser()
    define_parser(parser)
    args = parser.parse_args()

    ## When running with the specified seed 
    if len(args.seed_list_in) != 0:
        modify_seed_list(args)
        
    print('[args]')
    print(args)
    print()

    if not args.filename == "":
        ## now datetime
        now_date = datetime.now()
        now_md = now_date.strftime('%m%d')
        fine_name = args.dataset + '_' + args.filename
        fname = os.path.join('finetune', args.ft_type, fine_name)

        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
    
    train_results, val_results, test_results = np.array([]), np.array([]), np.array([])
    
    for i in range(args.run_cv):
        if len(args.seed_list) != 0: ## When running with the specified seed 
            try:
                args.runseed = args.seed_list[i]

            except:
                args.runseed = -1 ## If fewer than the number of iterations are entered, the seed will be replaced with a random seed
        else:
            args.runseed = -1 ## random seed
        
        if args.runseed == -1:
            print(f'================== {i+1} exec - RANDOM seed ==================')
        else:
            print(f'================== {i+1} exec - data seed: {args.seed}, run seed: {args.runseed} ==================')
        
        train_result, val_result, test_result, epoch, task_type = exec_main(args)
        print("[BEST - Epoch: %d] train: %.6f val: %.6f test: %.6f" % (epoch, train_result, val_result, test_result))

        if task_type == 'cls':
            train_result = train_result * 100
            val_result = val_result * 100
            test_result = test_result * 100


        if not args.filename == "":
            result_df = pd.DataFrame({'seed': [args.runseed],
                                      'epoch':[epoch],
                                      'train':[train_result],
                                      'valid':[val_result],
                                      'test':[test_result]})
            
            if not os.path.exists(fname+'_result.csv'):
                result_df.to_csv(fname+'_result.csv', mode='w', index=False)
            else:
                result_df.to_csv(fname+'_result.csv', mode='a', index=False, header=False)
        
        train_results = np.append(train_results, train_result)
        val_results = np.append(val_results, val_result)
        test_results = np.append(test_results, test_result)
        print()
    
    print(f'================== {args.run_cv} iteration Result: ==================')
    print('[test_results]')
    print(test_results)
    print('----------------------------------------------------------------------')
    print('[mean]')
    if task_type == 'cls':
        print("train: %.1f val: %.1f test: %.1f" % (train_results.mean(), val_results.mean(), test_results.mean()))
    elif task_type == 'reg':
        print("train: %.6f val: %.6f test: %.6f" % (train_results.mean(), val_results.mean(), test_results.mean()))
    print('[std]')
    print("train: %.2f val: %.2f test: %.2f" % (train_results.std(), val_results.std(), test_results.std()))
    


if __name__ == "__main__":
    main()
