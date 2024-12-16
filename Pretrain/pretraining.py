import warnings; warnings.filterwarnings('ignore') ## warning 
import argparse
from functools import partial

from splitters import pretrain_random_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import GNN

from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from tensorboardX import SummaryWriter

from datetime import datetime ##
from pathlib import Path
import os ##

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def train_more(args, model_list, loader, optimizer_list, device, alpha_l=1.0, loss_fn="sce"):
    model, dec_pred_atoms, dec_pred_maccs, dec_pred_descriptor, dec_pred_dist = model_list ##
    optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_maccs, optimizer_dec_pred_descriptor, optimizer_dec_pred_dist = optimizer_list ##
        
    model.train()
    loss_accum = 0 

    # ---------- Node-level ---------- #
    loss_ae_total = None; loss_lambda_ae_total = None
    if args.node_mode:
        dec_pred_atoms.train()
        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            criterion = nn.CrossEntropyLoss()
        loss_ae_total = 0; loss_lambda_ae_total = 0
    # ---------- Node-level ---------- #
        
    
    # ---------- Subgraph-level ---------- #
    loss_maccs_total = None; loss_alpha_maccs_total = None
    if args.maccs_mode:
        dec_pred_maccs.train()
        criterion_maccs = nn.BCEWithLogitsLoss()
        loss_maccs_total = 0; loss_alpha_maccs_total = 0
    # ---------- Subgraph-level ---------- #
    
    # ---------- Graph-level ---------- #
    loss_descriptor_total = None; loss_beta_descriptor_total = None    
    if args.descriptor_mode:
        dec_pred_descriptor.train()
        criterion_des = nn.MSELoss()
        loss_descriptor_total = 0; loss_beta_descriptor_total = 0
    # ---------- Graph-level ---------- #

    # ---------- 3D-level ---------- #
    loss_dist_total = None; loss_gamma_dist_total = None
    if args.dist_mode:
        dec_pred_dist.train()
        criterion_dist = nn.MSELoss(reduction='none')
        loss_dist_total = 0; loss_gamma_dist_total = 0
    # ---------- 3D-level ---------- #

    epoch_iter = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(epoch_iter):
        batch = batch.to(device)
        
        # Encoder
        if args.node_mode:
            node_attr_label = batch.node_attr_label
            masked_node_indices = batch.masked_atom_indices
            node_rep = model(batch.mask_node, batch.edge_index, batch.edge_attr) ## mask_node
        else:
            node_rep = model(batch.x, batch.edge_index, batch.edge_attr)

        # ---------- Node-level ---------- #
        loss_ae = 0
        if args.node_mode:
            pred_node = dec_pred_atoms(node_rep, batch.edge_index, batch.edge_attr, masked_node_indices)
            if loss_fn == "sce":
                loss_ae = criterion(node_attr_label, pred_node[masked_node_indices])
            else:
                loss_ae = criterion(pred_node.double()[masked_node_indices], batch.mask_node_label[:,0])
        # ---------- Node-level ---------- #
        
        # ---------- Subgraph-level ---------- #
        loss_maccs = 0
        if args.maccs_mode:
            pred_maccs = dec_pred_maccs(node_rep, batch.batch)
            loss_maccs = criterion_maccs(pred_maccs, batch.maccs.to(torch.float))
        # ---------- Subgraph-level ---------- #
        
        # ---------- Graph-level ---------- #
        loss_descriptor = 0
        if args.descriptor_mode:
            pred_descriptor = dec_pred_descriptor(node_rep, batch.batch)
            loss_descriptor = criterion_des(pred_descriptor, batch.descriptors_ss)
        # ---------- Graph-level ---------- #
        
        # ---------- 3D-level ---------- #
        loss_dist = 0
        if args.dist_mode:
            pred_dist = dec_pred_dist(node_rep)
            true_dist = torch.cdist(batch.pos, batch.pos)
            
            loss_dist = criterion_dist(pred_dist.to(torch.float32), true_dist.to(torch.float32))
            mask = torch.triu(torch.ones_like(loss_dist), diagonal=1).bool()  ## Diagonal matrix processing
            loss_dist = loss_dist[mask].mean()  ## Diagonal matrix processing
        # ---------- 3D-level ---------- #
            
        loss = (args.ae_lambda * loss_ae) + (args.maccs_alpha * loss_maccs) + (args.des_beta * loss_descriptor) + (args.dist_gamma * loss_dist)

        optimizer_model.zero_grad()
        if args.node_mode: ## Node-level
            optimizer_dec_pred_atoms.zero_grad()
        if args.maccs_mode: ## Subgraph-level
            optimizer_dec_pred_maccs.zero_grad()
        if args.descriptor_mode: ## Graph-level
            optimizer_dec_pred_descriptor.zero_grad()
        if args.dist_mode: ## 3D-level
            optimizer_dec_pred_dist.zero_grad()

        loss.backward()

        optimizer_model.step()
        if args.node_mode: ## Node-level
            optimizer_dec_pred_atoms.step()
        if args.maccs_mode: ## Subgraph-level
            optimizer_dec_pred_maccs.step()
        if args.descriptor_mode: ## Graph-level
            optimizer_dec_pred_descriptor.step()
        if args.dist_mode: ## 3D-level
            optimizer_dec_pred_dist.step()

        loss_accum += float(loss.cpu().item())
        epoch_iter.set_description(f"train_loss: {loss.item():.4f}")
        
        ## addition
        if args.node_mode: ## Node-level
            loss_ae_total += float(loss_ae.cpu().item())
            loss_lambda_ae_total += float((args.ae_lambda * loss_ae).cpu().item())
        
        if args.maccs_mode: ## Subgraph-level
            loss_maccs_total += float(loss_maccs.cpu().item())
            loss_alpha_maccs_total += float((args.maccs_alpha * loss_maccs).cpu().item())
            
        if args.descriptor_mode: ## Graph-level
            loss_descriptor_total += float(loss_descriptor.cpu().item())
            loss_beta_descriptor_total += float((args.des_beta * loss_descriptor).cpu().item())
            
        if args.dist_mode: ## 3D-level
            loss_dist_total += float(loss_dist.cpu().item())
            loss_gamma_dist_total += float((args.dist_gamma * loss_dist).cpu().item())

    
    if args.node_mode:
        loss_ae_total = loss_ae_total/(step+1)
        loss_lambda_ae_total = loss_lambda_ae_total/(step+1)
    
    if args.maccs_mode:
        loss_maccs_total = loss_maccs_total/(step+1)
        loss_alpha_maccs_total = loss_alpha_maccs_total/(step+1)
        
    if args.descriptor_mode:
        loss_descriptor_total = loss_descriptor_total/(step+1)
        loss_beta_descriptor_total = loss_beta_descriptor_total/(step+1)
        
    if args.dist_mode:
        loss_dist_total = loss_dist_total/(step+1)
        loss_gamma_dist_total = loss_gamma_dist_total/(step+1)
        
    return loss_accum/(step+1), loss_ae_total, loss_lambda_ae_total, loss_maccs_total, loss_alpha_maccs_total, loss_descriptor_total, loss_beta_descriptor_total, loss_dist_total, loss_gamma_dist_total

def eval(args, model_list, loader, device, alpha_l=1.0, loss_fn="sce"): 
    model, dec_pred_atoms, dec_pred_maccs, dec_pred_descriptor, dec_pred_dist = model_list ##
        
    model.eval()    
    loss_accum = 0 

    # ---------- Node-level ---------- #
    loss_ae_total = None
    if args.node_mode:
        dec_pred_atoms.eval()
        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            criterion = nn.CrossEntropyLoss()
        loss_ae_total = 0
    # ---------- Node-level ---------- #

    # ---------- Subgraph-level ---------- #
    loss_maccs_total = None
    if args.maccs_mode:
        dec_pred_maccs.eval()
        criterion_maccs = nn.BCEWithLogitsLoss()
        loss_maccs_total = 0
    # ---------- Subgraph-level ---------- #
    
    # ---------- Graph-level ---------- #
    loss_descriptor_total = None
    if args.descriptor_mode:
        dec_pred_descriptor.eval()
        criterion_des = nn.MSELoss()
        loss_descriptor_total = 0
    # ---------- Graph-level ---------- #
        
    # ---------- 3D-level ---------- #
    loss_dist_total = None
    if args.dist_mode:
        dec_pred_dist.train()
        criterion_dist = nn.MSELoss(reduction='none')
        loss_dist_total = 0
    # ---------- 3D-level ---------- #


    epoch_iter = tqdm(loader, desc="Iteration_eval")
    for step, batch in enumerate(epoch_iter):
        batch = batch.to(device)            
        
        # Encoder
        with torch.no_grad():
            if args.node_mode:
                node_attr_label = batch.node_attr_label
                masked_node_indices = batch.masked_atom_indices
                node_rep = model(batch.mask_node, batch.edge_index, batch.edge_attr)
            else:
                node_rep = model(batch.x, batch.edge_index, batch.edge_attr)

        # ---------- Node-level ---------- #
        loss_ae = 0
        if args.node_mode:
            with torch.no_grad():
                pred_node = dec_pred_atoms(node_rep, batch.edge_index, batch.edge_attr, masked_node_indices)
            if loss_fn == "sce":
                loss_ae = criterion(node_attr_label, pred_node[masked_node_indices])
            else:
                loss_ae = criterion(pred_node.double()[masked_node_indices], batch.mask_node_label[:,0])
        # ---------- Node-level ---------- #
        
        # ---------- Subgraph-level ---------- #
        loss_maccs = 0
        if args.maccs_mode:
            with torch.no_grad():
                pred_maccs = dec_pred_maccs(node_rep, batch.batch)
            loss_maccs = criterion_maccs(pred_maccs, batch.maccs.to(torch.float))
        # ---------- Subgraph-level ---------- #
        
        # ---------- Graph-level ---------- #
        loss_descriptor = 0
        if args.descriptor_mode:
            with torch.no_grad():
                pred_descriptor = dec_pred_descriptor(node_rep, batch.batch)
            loss_descriptor = criterion_des(pred_descriptor, batch.descriptors_ss)
        # ---------- Graph-level ---------- #
        
        # ---------- 3D-level ---------- #
        loss_dist = 0
        if args.dist_mode:
            with torch.no_grad():
                pred_dist = dec_pred_dist(node_rep)
            true_dist = torch.cdist(batch.pos, batch.pos)
            
            loss_dist = criterion_dist(pred_dist.to(torch.float32), true_dist.to(torch.float32))
            mask = torch.triu(torch.ones_like(loss_dist), diagonal=1).bool()  ## Diagonal matrix processing
            loss_dist = loss_dist[mask].mean()  ## Diagonal matrix processing
        # ---------- 3D-level ---------- #
            
        loss = (args.ae_lambda * loss_ae) + (args.maccs_alpha * loss_maccs) + (args.des_beta * loss_descriptor) + (args.dist_gamma * loss_dist)

        loss_accum += float(loss.cpu().item())
        epoch_iter.set_description(f"valid_loss: {loss.item():.4f}")
        
        ## addition
        if args.node_mode: ## Node-level
            loss_ae_total += float(loss_ae.cpu().item())
        
        if args.maccs_mode: ## Subgraph-level
            loss_maccs_total += float(loss_maccs.cpu().item())
                        
        if args.descriptor_mode: ## Graph-level
            loss_descriptor_total += float(loss_descriptor.cpu().item())
            
        if args.dist_mode: ## 3D-level
            loss_dist_total += float(loss_dist.cpu().item())

    if args.node_mode:
        loss_ae_total = loss_ae_total/(step+1)
    
    if args.maccs_mode:
        loss_maccs_total = loss_maccs_total/(step+1)
        
    if args.descriptor_mode:
        loss_descriptor_total = loss_descriptor_total/(step+1)
    
    if args.dist_mode:
        loss_dist_total = loss_dist_total/(step+1)
        
    return loss_accum/(step+1), loss_ae_total, loss_maccs_total, loss_descriptor_total, loss_dist_total
    
    
def main():    
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    # Training settings
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.25,
                        help='dropout ratio (default: 0.25)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent', 
                        help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default = '', 
                        help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=42, 
                        help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='number of workers for dataset loading')
    parser.add_argument('--input_model_file', type=str, default=None)
    parser.add_argument("--alpha_l", type=float, default=1.0)
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--decoder", type=str, default="gin")
    parser.add_argument("--use_scheduler", action="store_true", default=False)
    
    ## addition
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--run_seed', type=int, default=-1)
    
    ## Node-level
    parser.add_argument('--node_mode', action="store_true", default=False, help='node-level pretext task')
    parser.add_argument('--ae_lambda', type=float, default=1.0, help='lambda_1 (node-level)')
    parser.add_argument('--dec_layer', type=int, default=1, help='the number of node-level decoder layer')

    ## Subgraph-level
    parser.add_argument('--maccs_mode', action="store_true", default=False, help='subraph-level pretext task')
    parser.add_argument('--maccs_alpha', type=float, default=1.0, help='lambda_2 (subgraph-level)')
    parser.add_argument('--maccs_decay', type=float, default=0, help='weight decay at subgraph-level (default: 0)')
    parser.add_argument('--maccs_drop', type=float, default=0.0, help='dropout ratio at subgraph-level (default: 0)')
    
    ## Graph-level
    parser.add_argument('--descriptor_mode', action="store_true", default=False, help='graph-level pretext task')
    parser.add_argument('--des_beta', type=float, default=1.0, help='lambda_3 (graph-level)')
    parser.add_argument('--des_decay', type=float, default=0, help='weight decay at graph-level (default: 0)')
    parser.add_argument('--des_drop', type=float, default=0.0, help='dropout ratio at graph-level (default: 0)')
    
    
    parser.add_argument('--dist_mode', action="store_true", default=False, help='3D-level pretext task')
    parser.add_argument('--dist_dim', type=int, default=30, help='last embedding dimension')
    parser.add_argument('--dist_emb', type=int, default=256, help='hidden embedding dimension')
    parser.add_argument('--dist_gamma', type=float, default=1.0, help='lambda_4 (3D-level)')
    parser.add_argument('--dist_decay', type=float, default=0, help='weight decay at 3D-level (default: 0)')
    parser.add_argument('--dist_drop', type=float, default=0.0, help='dropout ratio at 3D-level (default: 0)')
    
    args = parser.parse_args()
    print(args)

    if (args.node_mode == False) and (args.maccs_mode == False) and (args.descriptor_mode == False) and (args.dist_mode == False):
        raise ValueError("At least one pretext task must be executed.") 

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if args.run_seed != -1:
        torch.manual_seed(args.run_seed)
        np.random.seed(args.run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.run_seed)

    if args.node_mode:
        print("num layer: %d mask rate: %f mask edge: %d" %(args.num_layer, args.mask_rate, args.mask_edge))
        print()

    # set up dataset and transform function.
    print("### set up dataset and transform function.") ##
    dataset_name = args.dataset

    if args.dist_mode:
        from loader_conf import MoleculeDataset ##
        dataset = MoleculeDataset("dataset_conf/" + dataset_name, dataset=dataset_name)
    else:
        from loader_conf import MoleculeDataset ##
        dataset = MoleculeDataset("dataset_info/" + dataset_name, dataset=dataset_name)
    
    ########## print ##########
    print('[dataset] - MoleculeDataset')
    print(dataset)
    print(dataset.data)
    ########## print ##########
    
    ########## split train and valid ##########
    train_dataset, val_dataset = pretrain_random_split(dataset, null_value=0, frac_train=0.9, frac_valid=0.1, seed=args.seed)
    print(f'train: {train_dataset}, valid: {val_dataset}')
    
    if args.node_mode and args.dist_mode:
        from dataloader import DataLoaderMasking3dDistPred
        train_loader = DataLoaderMasking3dDistPred(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, mask_rate=args.mask_rate, mask_edge=args.mask_edge)
        val_loader = DataLoaderMasking3dDistPred(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, mask_rate=args.mask_rate, mask_edge=args.mask_edge)

    elif args.node_mode:
        from dataloader import DataLoaderMaskingPred
        train_loader = DataLoaderMaskingPred(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, mask_rate=args.mask_rate, mask_edge=args.mask_edge)
        val_loader = DataLoaderMaskingPred(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, mask_rate=args.mask_rate, mask_edge=args.mask_edge)

    elif args.dist_mode:
        from dataloader import DataLoaderDist3DPred
        train_loader = DataLoaderDist3DPred(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoaderDist3DPred(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    else:
        from torch_geometric.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    ########## split train and valid ##########
    print()
    
    # set up models, one for pre-training and one for context embeddings
    print("### set up models, one for pre-training and one for context embeddings")
    
    # Encoder
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device) ## Encoder

    if args.input_model_file is not None and args.input_model_file != "":
        model.load_state_dict(torch.load(args.input_model_file))
        print("Resume training from:", args.input_model_file)

    # ---------- Node-level ---------- #
    atom_pred_decoder  = None
    NUM_NODE_ATTR = 119 # + 3
    if args.node_mode:
        from model import GNNDecoders
        atom_pred_decoder = GNNDecoders(hidden_dim=args.emb_dim, out_dim=NUM_NODE_ATTR, JK=args.JK, gnn_type=args.gnn_type, num_layer=args.dec_layer).to(device)
    # ---------- Node-level ---------- #
        
    # ---------- Subgraph-level ---------- #
    maccs_pred_decoder = None
    if args.maccs_mode:
        from model import MACCSPredictor
        maccs_dim = dataset.data['maccs'].shape[1] ## output dim
        maccs_pred_decoder = MACCSPredictor(hidden_dim=args.emb_dim, out_dim=maccs_dim, args=args).to(device)
    # ---------- Subgraph-level ---------- #
    
    # ---------- Graph-level ---------- #
    desciptor_pred_decoder = None
    if args.descriptor_mode:
        from model import DescriptorPredictor
        des_dim = dataset.data['descriptors_ss'].shape[1] ## output dim
        desciptor_pred_decoder = DescriptorPredictor(hidden_dim=args.emb_dim, out_dim=des_dim, args=args).to(device)
    # ---------- Graph-level ---------- #
    
    # ---------- 3D-level ---------- #
    dist_pred_decoder = None
    if args.dist_mode:
        from model import Decoder_DistPreds
        dist_pred_decoder = Decoder_DistPreds(in_dim=args.emb_dim, hidden_dim=args.dist_emb, out_dim=args.dist_dim, args=args).to(device)
    # ---------- 3D-level ---------- #

    model_list = [model, atom_pred_decoder, maccs_pred_decoder, desciptor_pred_decoder, dist_pred_decoder]
    
    print('[Model Architecture]')
    for idx, model_print in enumerate(model_list):
        print(f'model {idx+1}')
        print(model_print)
    print()

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    # ---------- Node-level ---------- #
    optimizer_dec_pred_atoms = None
    if args.node_mode:
        optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    # ---------- Node-level---------- #
    
    # ---------- Subgraph-level ---------- #
    optimizer_maccs_pred = None
    if args.maccs_mode:
        optimizer_maccs_pred = optim.Adam(maccs_pred_decoder.parameters(), lr=args.lr, weight_decay=args.maccs_decay)
    # ---------- Subgraph-level---------- #
    
    # ---------- Graph-level ---------- #
    optimizer_descriptor_pred = None
    if args.descriptor_mode:
        optimizer_descriptor_pred = optim.Adam(desciptor_pred_decoder.parameters(), lr=args.lr, weight_decay=args.des_decay)
    # ---------- Graph-level ---------- #
    
    # ---------- 3D-level ---------- #
    optimizer_dist_pred = None
    if args.dist_mode:
        optimizer_dist_pred = optim.Adam(dist_pred_decoder.parameters(), lr=args.lr, weight_decay=args.dist_decay)
    # ---------- 3D-level ---------- #

    optimizer_list = [optimizer_model, optimizer_dec_pred_atoms, optimizer_maccs_pred, optimizer_descriptor_pred, optimizer_dist_pred]
    print('[Optimizer]')
    for idx, optim_print in enumerate(optimizer_list):
        print(f'Optimizer {idx+1}')
        print(optim_print)
    print()

    # ---------- output file modify ---------- #
    now_md = datetime.now().strftime("%m%d")
    output_file_temp = "./checkpoints/" + f"{args.gnn_type}" + f"_{args.dataset}_" + args.output_model_file + f"_{now_md}"
    path_tmp = Path(output_file_temp); path_tmp.mkdir(parents=True, exist_ok=True) ## mkdir
    writer = SummaryWriter(output_file_temp) ## tensorboard
    # ---------- output file modify ---------- #
        
    for epoch in range(1, args.epochs+1):
        print(f'==== Epoch {epoch} - Node-level {args.node_mode} || Subgraph-level {args.maccs_mode} || Graph-level {args.descriptor_mode} || 3D-level {args.dist_mode}')
        
        train_loss, train_ae_loss, train_ae_lambda_loss, train_maccs_loss, train_maccs_alpha_loss, train_des_loss, train_des_beta_loss, train_dist_loss, train_dist_gamma_loss = train_more(args, model_list, train_loader, optimizer_list, device)
        val_loss, val_ae_loss, val_maccs_loss, val_des_loss, val_dist_loss = eval(args, model_list, val_loader, device)
        
        torch.save(model.state_dict(), os.path.join(output_file_temp, f"model_epoch_{epoch}_ValLoss_{val_loss}.pth")) ## modify
            
        print(f"[TRAIN] Loss: {train_loss}")
        print(f"[VALID] Loss: {val_loss}")
        
        writer.add_scalar('Train/Loss', train_loss, epoch) ##
        writer.add_scalar('Valid/Loss', val_loss, epoch) ##
        

        if args.node_mode:
            print("[TRAIN NODE-level] NODE Loss: %.6f  NODE_lambda Loss: %.6f" % (train_ae_loss, train_ae_lambda_loss)) ##
            print("[VALID NODE-level] NODE Loss: %.6f" % (val_ae_loss)) ##
            
            writer.add_scalar('Train/NODE_Loss', train_ae_loss, epoch) ##
            writer.add_scalar('Train/NODE_lambda', train_ae_lambda_loss, epoch) ##
            writer.add_scalar('Valid/NODE_Loss', val_ae_loss, epoch) ##
        
        
        if args.maccs_mode:
            print("[TRAIN Subgraph-level] Subgraph Loss: %.6f  Subgraph_lambda Loss: %.6f" % (train_maccs_loss, train_maccs_alpha_loss)) ##
            print("[VALID Subgraph-level] Subgraph Loss: %.6f" % (val_maccs_loss)) ##
            
            writer.add_scalar('Train/Subgraph_Loss', train_maccs_loss, epoch) ##
            writer.add_scalar('Train/Subgraph_lambda', train_maccs_alpha_loss, epoch) ##
            writer.add_scalar('Valid/Subgraph_Loss', val_maccs_loss, epoch) ##
            
            
        if args.descriptor_mode:
            print("[TRAIN Graph-level] Graph Loss: %.6f  Graph_lambda Loss: %.6f" % (train_des_loss, train_des_beta_loss)) ##
            print("[VALID Graph-level] Graph Loss: %.6f" % (val_des_loss)) ##
            
            writer.add_scalar('Train/Graph_Loss', train_des_loss, epoch) ##
            writer.add_scalar('Train/Graph_lambda', train_des_beta_loss, epoch) ##
            writer.add_scalar('Valid/Graph_Loss', val_des_loss, epoch) ##
            
            
        if args.dist_mode:
            print("[TRAIN 3D] 3D Loss: %.6f  3D_lambda Loss: %.6f" % (train_dist_loss, train_dist_gamma_loss)) ##
            print("[VALID 3D] 3D Loss: %.6f" % (val_dist_loss)) ##
            
            writer.add_scalar('Train/3D_Loss', train_dist_loss, epoch) ##
            writer.add_scalar('Train/3D_lambda', train_dist_gamma_loss, epoch) ##
            writer.add_scalar('Valid/3D_Loss', val_dist_loss, epoch) ##
        
        print()

    writer.close()
    
if __name__ == "__main__":
    main()
