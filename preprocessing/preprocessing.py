import numpy as np
import xarray as xr
import pickle
import time
import argparse
import sys
import torch
import netCDF4 as nc
from torch_geometric.utils import degree

from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
transform = T.AddLaplacianEigenvectorPE(k=2)

from utils.tools import write_log
from utils.graph import cut_window, retain_valid_nodes, derive_edge_index_within, derive_edge_index_multiscale

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path_phase_2', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str)
parser.add_argument('--log_file', type=str)
parser.add_argument('--input_path_gripho', type=str)
parser.add_argument('--input_path_topo', type=str)
parser.add_argument('--gripho_file', type=str)
parser.add_argument('--topo_file', type=str)
parser.add_argument('--land_use_path', type=str)
parser.add_argument('--land_use_file', type=str)

#-- lat lon grid values
parser.add_argument('--lon_min', type=float)
parser.add_argument('--lon_max', type=float)
parser.add_argument('--lat_min', type=float)
parser.add_argument('--lat_max', type=float)
parser.add_argument('--lon_grid_radius_high', type=float)
parser.add_argument('--lat_grid_radius_high', type=float)
parser.add_argument('--lon_grid_radius_low', type=float, default=0.36)
parser.add_argument('--lat_grid_radius_low', type=float, default=0.36)

#-- other
parser.add_argument('--suffix_phase_2', type=str, default='')
parser.add_argument('--mask_path', type=str)
parser.add_argument('--mask_file', type=str)
parser.add_argument('--predictors_type', type=str)
parser.add_argument('--target_multiplier', type=float, default=1)

#-- era5
parser.add_argument('--input_files_prefix_low', type=str, help='prefix for the input files (convenction: {prefix}{parameter}.nc)', default='')
parser.add_argument('--n_levels_low', type=int, help='number of pressure levels considered', default=5)


######################################################
##------------- PRELIMINARY OPERATIONS -------------##
######################################################

args = parser.parse_args()

write_log("\nStart!", args, accelerator=None, mode='w')

time_start = time.time()

######################################################
##-------------------- PHASE 2A --------------------##
##---------- PREPROCESSING LOW RES DATA ------------##
######################################################

if args.predictors_type == "era5":
    params = ['q', 't', 'u', 'v', 'z']
elif args.predictors_type == "regcm":
    params = ['hus', 'ta', 'ua', 'va', 'zg']
else:
    raise Exception("args.predictors_type should be either era5 or regcm")

n_params = len(params)

#-------------------------#
# INPUT TENSOR FROM FILES #
#-------------------------#

write_log('\nStarting the preprocessing of the low resolution data.', args, accelerator=None, mode='a')

for p_idx, p in enumerate(params):
    if args.predictors_type == "era5":
        write_log(f'\nPreprocessing {args.input_files_prefix_low}{p}.nc ...', args, accelerator=None, mode='a')
        with nc.Dataset(f'{args.input_path_phase_2}{args.input_files_prefix_low}{p}.nc') as ds:
            data = ds[p][:]
            if p_idx == 0: # first parameter being processed -> get dimensions and initialize the input dataset
                try:
                    lat_low = ds['latitude'][:]
                    lon_low = ds['longitude'][:]
                except:
                    lat_low = ds['lat'][:]
                    lon_low = ds['lon'][:]
                lat_dim = len(lat_low)
                lon_dim = len(lon_low)
                time_dim = len(ds['time'])
                input_ds = np.zeros((time_dim, n_params, args.n_levels_low, lat_dim, lon_dim), dtype=np.float32) # time, variables, levels, lat, lon
        input_ds[:, p_idx,:,:,:] = data
            
    elif args.predictors_type == "regcm":
        with nc.Dataset(f'{args.input_path_phase_2}{args.input_files_prefix_low}{p}.nc') as ds:
            for l_idx, level in enumerate(['200', '500', '700', '850', '1000']):
                write_log(f'\nPreprocessing {args.input_files_prefix_low}{p}.nc for level {level}', args, accelerator=None, mode='a')
                var_name = f"{p}{level}"
                _data = ds[var_name][:]
                if "zg" in var_name:
                    _data *= 9.81
                    write_log(f'\nMultiplying {var_name} by 9.81 to get kg*m^2/s^2.', args, accelerator=None, mode='a')
                if p_idx == 0 and l_idx == 0: # first parameter being processed -> get dimensions and initialize the input dataset
                    try:
                        lat_low = ds['latitude'][:]
                        lon_low = ds['longitude'][:]
                    except:
                        lat_low = ds['lat'][:]
                        lon_low = ds['lon'][:]
                    lat_dim = len(lat_low)
                    lon_dim = len(lon_low)
                    time_dim = len(ds['time'])
                    input_ds = np.zeros((time_dim, n_params, args.n_levels_low, lat_dim, lon_dim), dtype=np.float32) # time, variables, levels, lat, lon
                data = torch.from_numpy(_data)
                mask = torch.from_numpy(_data.mask.astype(bool))
                data[mask] = torch.nan
                input_ds[:, p_idx,l_idx,:,:] = data.numpy()

lat_low = np.flip(lat_low, axis=0)  # Flip the latitude array along the first axis
lat_low, lon_low = np.meshgrid(lat_low, lon_low, indexing='ij')

lat_low = lat_low.flatten()
lon_low = lon_low.flatten()

#--------------------------#
# POST-PROCESSING OF INPUT #
#--------------------------#

input_ds = torch.tensor(input_ds)
    
#----- Flip the dataset -----#
# the origin in the input files is in the top left corner, while we use the bottom left corner    
input_ds = torch.flip(input_ds, [3])

#### IMPORTANT CHANGE - NORMALIZATION NOW IN MAIN AND PREDICTION #### 
 
input_ds = torch.permute(input_ds, (3,4,0,1,2)) # lat, lon, time, vars, levels
input_ds = torch.flatten(input_ds, end_dim=1)   # num_nodes, time, vars, levels

with open(args.output_path + args.log_file, 'a') as f:
    f.write(f'\nPreprocessing of low resolution data finished.')


######################################################
##-------------------- PHASE 2B --------------------##    
##--------- PREPROCESSING HIGH RES DATA ------------##
######################################################

write_log(f"\n\nStarting the preprocessing of high resolution data.", args, accelerator=None, mode='a')

#-------------------------------#
# CUT LON, LAT, PR, Z TO WINDOW #
#-------------------------------#

write_log(f"\nLoading target and topography.", args, accelerator=None, mode='a')
dataset_high = xr.open_dataset(args.input_path_gripho + args.gripho_file)
topo = xr.open_dataset(args.input_path_topo + args.topo_file)

lon = dataset_high.lon.to_numpy()
lat = dataset_high.lat.to_numpy()
if lon.shape != lat.shape:
    lon, lat = np.meshgrid(lon, lat)
try:
    target_high = dataset_high.pr.to_numpy()
except:
    target_high = dataset_high.tp.to_numpy()

write_log(f"\nLoading mask_land.", args, accelerator=None, mode='a')

if args.predictors_type == "regcm":
    z = topo.orog.to_numpy()
    mask_land = xr.open_dataset(args.mask_path + args.mask_file)
    mask_land = mask_land.pr.to_numpy().squeeze()
    lon_z = topo.lon.to_numpy()
    lat_z = topo.lat.to_numpy()
else:
    z = topo.z.to_numpy()
    mask_land = xr.open_dataset(args.mask_path + args.mask_file)
    try:
        mask_land = mask_land.pr.to_numpy().squeeze()
    except:
        mask_land = mask_land.tp.to_numpy().squeeze()
    lon_z = topo.lon.to_numpy()
    lat_z = topo.lat.to_numpy()
if lon_z.shape != lat_z.shape:
    lon_z, lat_z = np.meshgrid(lon_z, lat_z)
    
if args.target_multiplier is not None:
    target_high *= args.target_multiplier
    write_log(f'\nMultiplying pr by {args.target_multiplier} to get the correct unit.', args, accelerator=None, mode='a')

# Reading LAND USE data
if args.land_use_path == "" or args.land_use_file == "":
    write_log(f"\nLand use file not provided, ignoring.", args, accelerator=None, mode='a')
else:
    write_log(f"\nLoading land use.", args, accelerator=None, mode='a')

    landU  = xr.open_dataset(args.land_use_path+args.land_use_file,  engine='netcdf4') #open nc file by default with netcdf4, if avail
    water = landU.water.to_numpy()
    coast = landU.coast.to_numpy()
    urban_MD = landU.urban_MD.to_numpy()
    urban_HD = landU.urban_HD.to_numpy()
    forest = landU.forest.to_numpy()
    ucrop = landU.ucrop.to_numpy()
    lon_landU = landU.lon.to_numpy()
    lat_landU = landU.lat.to_numpy()
    if lon_landU.shape != lat_landU.shape:
        lon_landU, lat_landU = np.meshgrid(lon_landU, lat_landU)

write_log("\nCutting the window...", args, accelerator=None, mode='a')

#-- Cut gripho and topo to the desired window --#
lon_high, lat_high, target_high = cut_window(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon, lat, target_high)

print("target done!")

if mask_land is not None:
    lon_high_z, lat_high_z, z_high, mask_land_high = cut_window(
            args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon_z, lat_z, z, mask_land)
else:
    lon_high_z, lat_high_z, z_high = cut_window(
            args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon_z, lat_z, z)
    mask_land_high = None

assert (np.allclose(lon_high, lon_high_z, atol=0.01) and lon_high.shape == lon_high_z.shape)

print("z done!")

if args.land_use_path != "" and args.land_use_file != "":
    lon_high_landU, lat_high_landU, water_high, coast_high, urban_MD_high, urban_HD_high, forest_high, ucrop_high = cut_window(
            args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon_landU, lat_landU, water, coast, urban_MD, urban_HD, forest, ucrop)

    assert (np.allclose(lon_high, lon_high_landU, atol=0.01) and lon_high.shape == lon_high_landU.shape)
    assert (np.allclose(lat_high, lat_high_z, atol=0.01) and lon_high.shape == lat_high_z.shape)
    assert (np.allclose(lat_high, lat_high_landU, atol=0.01) and lon_high.shape == lat_high_landU.shape)

    print("land use done!")

write_log(f"\nDone! Window is [{lon_high.min()}, {lon_high.max()}] x [{lat_high.min()}, {lat_high.max()}] with {target_high.shape[1]} nodes.", args, accelerator=None, mode='a')

write_log(f"\nlon shape {lon_high.shape}, lat shape {lat_high.shape}, pr shape {target_high.shape}, z shape {z_high.shape}", args, accelerator=None, mode='a')

#------------------------------------#
# REMOVE NODES NOT IN LAND TERRITORY #
#------------------------------------#

valid_nodes = retain_valid_nodes(target_high, mask_land_high)

# Apply the mask
target_high = target_high[:,valid_nodes]
lon_high = lon_high[valid_nodes]
lat_high = lat_high[valid_nodes]
z_high = z_high[valid_nodes]

target_high = target_high.swapaxes(0,1) # (num_nodes, time)

num_nodes_high = target_high.shape[0]

write_log(f"\nAfter removing the non land territory nodes, the high resolution graph has {num_nodes_high} nodes.", args, accelerator=None, mode='a')

if args.land_use_path != "" and args.land_use_file != "":
    water_high = water_high[valid_nodes]
    coast_high = coast_high[valid_nodes]
    urban_MD_high = urban_MD_high[valid_nodes]
    urban_HD_high = urban_HD_high[valid_nodes]
    forest_high = forest_high[valid_nodes]
    ucrop_high = ucrop_high[valid_nodes]

    land_vars_high = np.stack([water_high, coast_high, urban_MD_high, urban_HD_high, forest_high, ucrop_high], axis=-1)

#---------------------------------------#
# CLASSIFICATION AND REGRESSION TARGETS #
#---------------------------------------#

target_high = torch.tensor(target_high)

write_log("\nWriting some files...", args, accelerator=None, mode='a')

#-- WRITE THE FILES --#       
with open(args.output_path + 'pr_target.pkl', 'wb') as f:
    pickle.dump(target_high, f)

#### IMPORTANT CHANGE - NORMALIZATION NOW IN MAIN AND PREDICTION #### 

#-----------------#
# BUILD THE GRAPH #
#-----------------#

low_high_graph = HeteroData()
high_graph = Data()

#-- EDGES --#

edges_low2high, _ = derive_edge_index_multiscale(lon_senders=lon_low, lat_senders=lat_low,
                                lon_receivers=lon_high, lat_receivers=lat_high, k=9, undirected=False)

edges_high, _ = derive_edge_index_within(lon_radius=args.lon_grid_radius_high, lat_radius=args.lat_grid_radius_high,
                                lon_senders=lon_high, lat_senders=lat_high, lon_receivers=lon_high, lat_receivers=lat_high)

#-- TO GRAPH ATTRIBUTES --#

low_high_graph['low'].x = input_ds
low_high_graph['low'].lat = torch.tensor(lat_low)
low_high_graph['low'].lon = torch.tensor(lon_low)

low_high_graph['high'].lat = torch.tensor(lat_high)
low_high_graph['high'].lon = torch.tensor(lon_high)
low_high_graph['high'].z_std = torch.tensor(z_high).unsqueeze(-1)
if args.land_use_path != "" and args.land_use_file != "":
    low_high_graph['high'].land_std = torch.tensor(land_vars_high).float()
    low_high_graph['high'].x = torch.cat((low_high_graph['high'].z_std, low_high_graph['high'].land_std),dim=-1)
else:
    low_high_graph['high'].x = low_high_graph['high'].z_std

# High within High
low_high_graph['high', 'within', 'high'].edge_index = torch.tensor(edges_high)
low_high_graph['high', 'within', 'high'].edge_attr = torch.tensor(edges_high_attr).float()

# Low to High
low_high_graph['low', 'to', 'high'].edge_index = torch.tensor(edges_low2high)
low_high_graph['low', 'to', 'high'].edge_attr = torch.tensor(edges_low2high_attr).float()

#-- WRITE THE GRAPH --#

with open(args.output_path + 'low_high_graph' + args.suffix_phase_2 + '.pkl', 'wb') as f:
    pickle.dump(low_high_graph, f)

write_log(f"\nIn total, preprocessing took {time.time() - time_start} seconds", args, accelerator=None, mode='a')  
