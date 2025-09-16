import numpy as np
import pickle
import torch
import argparse
import time
import os
import matplotlib.pyplot as plt
import importlib

import safetensors

from accelerate import Accelerator

from torch_geometric.data import HeteroData
from torch_geometric.utils import degree

import dataset
from dataset import Dataset_Graph, Iterable_Graph

from utils.tools import date_to_idxs, set_seed_everything
from utils.train_test import Tester

from utils.plots import create_zones, extremes_cmap
from utils.plots import plot_maps, plot_single_map, plot_mean_time_series, plot_seasonal_maps
from utils.tools import date_to_idxs, write_log, standardize_input
        

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

parser.add_argument('--train_path_reg', type=str)
parser.add_argument('--train_path_cl', type=str)
parser.add_argument('--checkpoint_reg', type=str, default=None)
parser.add_argument('--checkpoint_cl', type=str, default=None)
parser.add_argument('--output_file', type=str, default="G_predictions.pkl")

parser.add_argument('--graph_file', type=str, default=None) 
parser.add_argument('--target_file', type=str, default="pr_target.pkl") 
parser.add_argument('--model_type', type=str, default=None)
parser.add_argument('--model', type=str, default=None) 
parser.add_argument('--dataset_name', type=str, default=None) 
parser.add_argument('--mode', type=str, default="cl_reg") 
parser.add_argument('--test_idxs_file', type=str, default="")
parser.add_argument('--stats_mode', type=str, default="var") 
parser.add_argument('--target_type', type=str, default="precipitation")
parser.add_argument('--seq_l', type=int, default=24)

#-- start and end training dates
parser.add_argument('--test_year_start', type=int)
parser.add_argument('--test_month_start', type=int)
parser.add_argument('--test_day_start', type=int)
parser.add_argument('--test_year_end', type=int)
parser.add_argument('--test_month_end', type=int)
parser.add_argument('--test_day_end', type=int)
parser.add_argument('--first_year', type=int)
parser.add_argument('--first_year_input', type=int)

parser.add_argument('--batch_size', type=int)
parser.add_argument('--seed', type=int)

parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')

parser.add_argument('--make_plots',  action='store_true')
parser.add_argument('--no-make_plots', dest='make_plots', action='store_false')


if __name__ == '__main__':

    args = parser.parse_args()
    
    # Set all seeds
    set_seed_everything(seed=args.seed)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    if args.use_accelerate is True:
        accelerator = Accelerator()
    else:
        accelerator = None

    write_log("Starting the testing...", args, accelerator, 'w')
    write_log(f"Cuda is available: {torch.cuda.is_available()}. There are {torch.cuda.device_count()} available GPUs.", args, accelerator, 'a')

    if args.test_idxs_file == "":
        test_start_idx, test_end_idx = date_to_idxs(args.test_year_start, args.test_month_start,
            args.test_day_start, args.test_year_end, args.test_month_end,
            args.test_day_end, args.first_year)        
        if test_start_idx < 24:
            test_start_idx = 24
        test_idxs = torch.tensor([*range(test_start_idx, test_end_idx)])
        write_log(f"\nUsing the provided start and end test times to derive the test idxs.", args, accelerator, 'a')
    else:
        with open(args.train_path_reg+args.test_idxs_file, 'rb') as f:
            test_idxs = pickle.load(f)
        write_log(f"Using the provided test idxs vector.", args, accelerator, 'a')

    # Load the precipitation target
    with open(args.input_path+args.target_file, 'rb') as f:
        pr_target = pickle.load(f)

    # Load the graph
    with open(args.input_path+args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)

    if "3h" in args.model_typel:
        pr_target = torch.stack([torch.mean(pr_target[:,t-2:t+1], dim=1) for t in range(pr_target.shape[1])]).swapaxes(0,1)
        test_idxs = test_idxs[::3]
        write_log(f"A 3h time resolution is considered.", args, accelerator, 'a')

    # Load the input data statistics used during training
    # (At the moment we assume that the same statistics has been used for
    # the regressor and classifier in the RC model case)
    with open(args.train_path_reg + "means_low.pkl", 'rb') as f:
        means_low = pickle.load(f)
    with open(args.train_path_reg + "stds_low.pkl", 'rb') as f:
        stds_low = pickle.load(f)
    with open(args.train_path_reg + "means_high.pkl", 'rb') as f:
        means_high = pickle.load(f)
    with open(args.train_path_reg + "stds_high.pkl", 'rb') as f:
        stds_high = pickle.load(f)

    # Standardizing the input data
    low_high_graph['low'].x, low_high_graph['high'].x = standardize_input(
        low_high_graph['low'].x, low_high_graph['high'].x, means_low, stds_low, means_high, stds_high, args, accelerator) # num_nodes, time, vars, levels
    
    vars_names = ['q', 't', 'u', 'v', 'z']
    levels = ['200', '500', '700', '850', '1000']
    if args.stats_mode == "var":
        for var in range(5):
            write_log(f"\nLow var {vars_names[var]}: mean={low_high_graph['low'].x[:,:,var,:].mean()}, std={low_high_graph['low'].x[:,:,var,:].std()}",
                      args, accelerator, 'a')
    elif args.stats_mode == "field":
        for var in range(5):
            for lev in range(5):
                write_log(f"\nLow var {vars_names[var]} lev {levels[lev]}: mean={low_high_graph[:,:,var,lev].mean()}, std={low_high_graph[:,:,var,lev].std()}",
                          args, accelerator, 'a')
    
    write_log(f"\nHigh z: mean={low_high_graph['high'].x[:,0].mean()}, std={low_high_graph['high'].x[:,0].std()}",
              args, accelerator, 'a')
    write_log(f"\nHigh land_use: mean={low_high_graph['high'].x[:,1:].mean()}, std={low_high_graph['high'].x[:,1:].std()}",
              args, accelerator, 'a')
    
    if args.target_type == "temperature":
        low_high_graph['low'].x = torch.cat((low_high_graph['low'].x[:,:,:1,:], low_high_graph['low'].x[:,:,2:,:]), dim=2)

    low_high_graph['low'].x = torch.flatten(low_high_graph['low'].x, start_dim=2, end_dim=-1)   # num_nodes, time, vars*levels

    Dataset_Graph = getattr(dataset, args.dataset_name)
    
    dataset_graph = Dataset_Graph(targets=None, graph=low_high_graph, model_name=args.model_typel, seq_l=args.seq_l)

    custom_collate_fn = getattr(dataset, 'custom_collate_fn_graph')
        
    sampler_graph = Iterable_Graph(dataset_graph=dataset_graph, shuffle=False, idxs_vector=test_idxs)
        
    dataloader = torch.utils.data.DataLoader(dataset_graph, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph, collate_fn=custom_collate_fn)

    model_file = importlib.import_module(f"models.{args.model_typel}")
    Model = getattr(model_file, args.model_typel)
    if args.model_type == "cl_reg":
        model_cl = Model(seq_l=args.seq_l+1)
        model_reg = Model(seq_l=args.seq_l+1)
    else:
        if args.target_type == "temperature":
            model = Model(h_in=4*5, h_hid=4*5, high_in=1)
        else:
            model = model = Model(seq_l=args.seq_l+1)

    if accelerator is None:
        if args.model_type == "cl_reg":
            checkpoint_cl = torch.load(args.train_path_cl+args.checkpoint_cl, map_location=torch.device('cpu'), weights_only=True)
            checkpoint_reg = torch.load(args.train_path_reg+args.checkpoint_reg, map_location=torch.device('cpu'), weights_only=True)
        else:
            checkpoint_reg = torch.load(args.train_path_reg+args.checkpoint_reg, map_location=torch.device('cpu'), weights_only=True)
        device = 'cpu'
    else:
        if args.model_type == "cl_reg":
            try:
                checkpoint_cl = torch.load(args.train_path_cl+args.checkpoint_cl+"/pytorch_model.bin", weights_only=True)
            except:
                checkpoint_cl = safetensors.torch.load_file(args.train_path_cl+args.checkpoint_cl+"/model.safetensors")
                torch.save(checkpoint_cl, args.train_path_cl+args.checkpoint_cl+"pytorch_model.bin")
            try:
                checkpoint_reg = torch.load(args.train_path_reg+args.checkpoint_reg+"/pytorch_model.bin", weights_only=True)
            except:
                checkpoint_reg = safetensors.torch.load_file(args.train_path_reg+args.checkpoint_reg+"/model.safetensors")
                torch.save(checkpoint_reg, args.train_path_reg+args.checkpoint_reg+"pytorch_model.bin")
        else:
            try:
                checkpoint_reg = torch.load(args.train_path_reg+args.checkpoint_reg+"/pytorch_model.bin", weights_only=True)
            except:
                checkpoint_reg = safetensors.torch.load_file(args.train_path_reg+args.checkpoint_reg+"/model.safetensors")
                torch.save(checkpoint_reg, args.train_path_reg+args.checkpoint_reg+"pytorch_model.bin")
        device = accelerator.device
    
    write_log("\nLoading state dict.", args, accelerator, 'a')
    if args.model_type == "cl_reg":
        model_cl.load_state_dict(checkpoint_cl)
        model_reg.load_state_dict(checkpoint_reg)
    else:
        model.load_state_dict(checkpoint_reg)

    if accelerator is not None:
        if args.model_type == "cl_reg":
            model_cl, model_reg, dataloader = accelerator.prepare(model_cl, model_reg, dataloader)
        else:
            model, dataloader = accelerator.prepare(model, dataloader)

    # write_log(f"\nStarting the test, from idx {test_start_idx} to idx {test_end_idx}.", args, accelerator, 'a')

    tester = Tester()

    start = time.time()

    if args.model_type == "RC":
        pr_R, pr_C, times = tester.test_RC(model_R, model_C, dataloader, args=args, accelerator=accelerator)
    elif args.model_type == "R":
        pr_R, times = tester.test(model, dataloader, args=args, accelerator=accelerator)
    elif args.model_type == "Rall":
        pr_Rall, times = tester.test(model, dataloader, args=args, accelerator=accelerator)
    elif args.model_type == "C":
        pr_C, times = tester.test(model, dataloader, args=args, accelerator=accelerator)
    else:
        raise Exception("mode should be: 'RC', 'R', 'C' or 'Rall'")

    end = time.time()

    # Create the pyg object
    data = HeteroData()

    if accelerator is not None:
        accelerator.wait_for_everyone()

        # Gather the values in *tensors* across all processes and concatenate them on the first dimension. Useful to
        # regroup the predictions from all processes when doing evaluation.

        times = accelerator.gather(times).squeeze()

        if args.model_type == "RC":
            pr_R = accelerator.gather(pr_R)
            pr_C = accelerator.gather(pr_C)
        elif args.model_type == "R":
            pr_R = accelerator.gather(pr_R)
        elif args.model_type == "Rall":
            pr_Rall = accelerator.gather(pr_R)
        elif args.model_type == "C":
            pr_C = accelerator.gather(pr_C)
    
    times, indices = torch.sort(times)
    if args.model_type == "RC":
        pr_R = accelerator.gather(pr_R).squeeze().swapaxes(0,1)[:,indices].cpu().numpy()
        pr_C = accelerator.gather(pr_C).squeeze().swapaxes(0,1)[:,indices].cpu().numpy()
    elif args.model_type == "R":
        pr_R = accelerator.gather(pr_R).squeeze().swapaxes(0,1)[:,indices].cpu().numpy()
    elif args.model_type == "Rall":
        pr_Rall = accelerator.gather(pr_R).squeeze().swapaxes(0,1)[:,indices].cpu().numpy()
    elif args.model_type == "C":
        pr_C = accelerator.gather(pr_C).squeeze().swapaxes(0,1)[:,indices].cpu().numpy()

    
    data.pr_target = pr_target[:,test_idxs].cpu().numpy()
    data.times = times.cpu().numpy()
    data["low"].lat = low_high_graph["low"].lat.cpu().numpy()
    data["low"].lon = low_high_graph["low"].lon.cpu().numpy()
    data["high"].lat = low_high_graph["high"].lat.cpu().numpy()
    data["high"].lon = low_high_graph["high"].lon.cpu().numpy()

    degree = degree(low_high_graph['high', 'within', 'high'].edge_index[0], low_high_graph['high'].num_nodes)
    data["high"].degree = degree.cpu().numpy()
    
    write_log(f"\nDone. Testing concluded in {end-start} seconds.\nWrite the files.", args, accelerator, 'a')

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.output_file, 'wb') as f:
            pickle.dump(data, f)