#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:41:54 2023

@author: midhunm
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import time
import csv
from os.path import isabs, join

def negtozero(arr):
    return np.where(arr < 0, 0, arr)

def logtrans(arr):
    return np.log10(arr + 1)

def r_logtrans(arr):
    return 10**(arr) - 1

def standardize(array):
    mean, stdv = np.nanmean(array), np.nanstd(array)
    return (array - mean)/stdv, mean, stdv

def minmaxnorm(array):
    return (array - np.nanmin(array))/(np.nanmax(array) - np.nanmin(array))

def divclip(array, c=1000):
    return np.clip((array / c), 0, 1) 

def logclip(array, c=1000, c_min=0, c_max=1):
    return np.clip(logtrans(array), c_min, c_max)

class TimeHistory(Callback):
    def __init__(self, filename):
        self.filename = filename

    def on_train_begin(self, logs={}):
        self.times = []
        self.filename = self.filename

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        epoch_time_end = time.time()
        epoch_time = epoch_time_end - self.epoch_time_start
        self.times.append(epoch_time)
        with open(self.filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch_time])
            
def load_inputs_target_pairs(inputs_channels: dict, target_channels: dict, static_channels: dict = None, data_path: str = None, as_dict=False):
    
    def check_shape_consistency(data_dict):
        shapes = [arr.shape for arr in data_dict.values()]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(f"Inconsistent shapes found: {shapes}")
    
    # Load input channels
    sel_inputs_channels = {}
    for channel, npy_path in inputs_channels.items():
        npy_full_path = npy_path if isabs(npy_path) else join(data_path, npy_path)
        print(f'\nLoading Inputs Channel ... {channel}: {npy_full_path}')
        sel_inputs_channels[channel] = np.load(npy_full_path)
        print(f'\tShape: {sel_inputs_channels[channel].shape}')
    
    # Check consistency and stack inputs
    check_shape_consistency(sel_inputs_channels)
    inputs = np.stack(list(sel_inputs_channels.values()), axis=3)

    print(f"\n\tFinished Processing Inputs Channels: {inputs.shape}")
    print('*'*100)
    
    # Load target channels
    sel_target_channels = {}
    for channel, npy_path in target_channels.items():
        npy_full_path = npy_path if isabs(npy_path) else join(data_path, npy_path)
        print(f'\nLoading Target Channel ... {channel}: {npy_full_path}')
        sel_target_channels[channel] = np.load(npy_full_path)
        print(f'\tShape: {sel_target_channels[channel].shape}')
    
    # Check consistency and stack targets
    check_shape_consistency(sel_target_channels)
    target = np.stack(list(sel_target_channels.values()), axis=3)

    print(f"\n\tFinished Processing Target Channels: {target.shape}")
    print('*'*100)
    
    if static_channels is not None:
        # Load static channels
        sel_static_channels = {}
        for channel, npy_path in static_channels.items():
            npy_full_path = npy_path if isabs(npy_path) else join(data_path, npy_path)
            print(f'\nLoading Static Channel ... {channel}: {npy_full_path}')
            sel_static_channels[channel] = np.load(npy_full_path)
            print(f'\tShape: {sel_static_channels[channel].shape}')
        
        # Check consistency and stack static channels
        check_shape_consistency(sel_static_channels)
        static = np.stack(list(sel_static_channels.values()), axis=3)

        print(f"\n\tFinished Processing Static Channels: {static.shape}")
        print('*'*100)
        
        result = {'inputs': inputs, 'static': static, 'target': target}
    else:
        result = {'inputs': inputs, 'target': target}
    
    return result if as_dict else tuple(result.values())

def train_val_split(X, y, S=None, train_bounds=None, val_bounds=None, test_only=False, test_bounds=12784):
    if train_bounds is None:
        train_bounds = np.concatenate([
            np.arange(366, 1827),
            np.arange(2192, 3653),
            np.arange(4018, 5479),
            np.arange(5844, 7305),
            np.arange(7671, 9132),
            np.arange(9497, 10598),
            np.arange(11322, 12784)
        ])
    
    if val_bounds is None:
        val_bounds = np.concatenate([
            np.arange(0, 366),
            np.arange(1827, 2192),
            np.arange(3653, 4018),
            np.arange(5479, 5844),
            np.arange(7305, 7671),
            np.arange(9132, 9497),
            np.arange(10598, 11322)
        ])
    
    if S is not None:
        
        if test_only:
            return X[test_bounds:], S[test_bounds:], y[test_bounds:] 
        
        else:
            return X[train_bounds], X[val_bounds], S[train_bounds], S[val_bounds], y[train_bounds], y[val_bounds]
    
    else:
        
        if test_only:
            return X[test_bounds:], y[test_bounds:] 

        else:
            return X[train_bounds], X[val_bounds], y[train_bounds], y[val_bounds]


def make_predictions(model, x_test, batch_size=32, loss_fn=None, thres=0.5):
    """
    Make predictions using a trained model.

    Args:
        model (tf.keras.Model): The trained model for prediction.
        x_test (ndarray): Test data features.
        batch_size (int): Batch size for prediction.
        loss_fn (str): Type of loss function used in the model.
        thres (float): Threshold for rainfall occurrence.

    Returns:
        xarray.Dataset: A dataset containing the predicted variable.

    """
    preds = model.predict(x_test, verbose=1, batch_size=batch_size)

    if loss_fn == "gamma":
        print("\nGenerating rainfall (log) from Gamma Distribution")
        scale = np.exp(preds[:,:,:,0])
        shape = np.exp(preds[:,:,:,1])
        prob = preds[:,:,:,-1]
        rainfall = (prob > thres) * scale * shape
    else:
        print("\nGenerating rainfall (log) from WMAE")
        rainfall = preds
    return rainfall


# def load_inputs_target_pairs(inputs_channels: dict, target_channels: dict, static_channels: dict = None, data_path: str = None, as_dict=False):
    
#     # Select Inputs Channels
#     sel_inputs_channels = {}
#     for channel, npy_path in inputs_channels.items():
#         npy_full_path = npy_path if data_path is None else f'{data_path}/{npy_path}'
#         print(f'\tLoading Inputs Channel ... {channel}: {npy_full_path}')
#         sel_inputs_channels[channel] = np.load(npy_full_path).squeeze()
#         print(f'\tShape: {sel_inputs_channels[channel].shape}')
    
#     # Stack the input arrays
#     inputs = np.stack(list(sel_inputs_channels.values()), axis=3)
#     print(f"Finish Process Inputs Channels: {inputs.shape}")
    
#     # Select Taget Channels
#     sel_target_channels = {}
#     for channel, npy_path in target_channels.items():
#         npy_full_path = npy_path if data_path is None else f'{data_path}/{npy_path}'
#         print(f'\tLoading Target Channel ... {channel}:  {npy_full_path}')
#         sel_target_channels[channel] = np.load(npy_full_path).squeeze()
#         print(f'\tShape: {sel_target_channels[channel].shape}')

#     # Stack the target arrays
#     target = np.stack(list(sel_target_channels.values()), axis=3)
#     print(f"Finish Process Target Channels: {target.shape}")
    
#     if static_channels is not None:
#         # Select Static Channels
#         sel_static_channels = {}
#         for channel, npy_path in static_channels.items():
#             npy_full_path = npy_path if data_path is None else f'{data_path}/{npy_path}'
#             print(f'\tLoading Static Channel ... {channel}: {npy_full_path}')
#             sel_static_channels[channel] = np.load(npy_full_path).squeeze()
#             print(f'\tShape: {sel_static_channels[channel].shape}')
        
#         # Stack the input arrays
#         static = np.stack(list(sel_static_channels.values()), axis=3)
#         print(f"Finish Process Static Channels: {static.shape}")
        
#         print(f'Inputs shape: {inputs.shape} & Static shape: {static.shape} & Target shape: {target.shape}')
        
#         if as_dict:
#             return{
#                 'inputs': inputs,
#                 'static': static,
#                 'target': target,
#                 }
#         else:
#             return inputs, static, target
    
#     else:
#         if as_dict:
#             return{
#                 'inputs': inputs,
#                 'target': target,
#                 }
#         else:
#             return inputs, target

# def sel_percentile_above(data_dict, mean_series_path=None, p=25, bound=None, for_val=False):
#     """ 
#     Select based on percentile indices
#     """
#     if mean_series_path is not None:
#         fsum = np.load(mean_series_path)
#         if for_val:
#             fsum = fsum[bound:]
#         else:
#             fsum = fsum[:bound]
#     else:
#         if bound is None:
#             target = data_dict['target']
#         elif for_val:
#             target = data_dict['target'][bound:]
#         else:
#             target = data_dict['target'][:bound]
#         fsum = np.nanmean(target, axis=(1, 2))

#     p_thresh = np.nanpercentile(fsum, p)
#     p_idx = np.where(fsum >= p_thresh)[0]

#     inputs = data_dict['inputs'][p_idx]
#     static = data_dict['static'][p_idx]
#     target = data_dict['target'][p_idx]

#     return {
#         'inputs': inputs, 
#         'static': static, 
#         'target': target
#         }

# def generate_data(inputs_channels: dict,
#                   static_channels: dict,
#                   target_channels: dict,
#                   mean_series_path: str,
#                   p=None, 
#                   bound=12419, 
#                   for_val=False
#                   ):
#     """
#     Data Loader for Phy-SRDRN
#     """
    
#     data_dict = load_inputs_target_pairs(inputs_channels, static_channels, target_channels)
    
#     if p is not None:
    
#         data_dict = sel_percentile_above(data_dict, mean_series_path, p, bound, for_val)
               
#     return data_dict['inputs'], data_dict['static'], data_dict['target']

# def interpolator(inp_arr, ups_factors):
#     # input layer
#     IN = x = tf.keras.layers.Input(input_size=inp_arr.shape[1:], name='unet_in')
#     for _, ups_size in enumerate(ups_factors):
#         x = tf.keras.layers.UpSampling2D(size=ups_size, interpolation='bilinear')(x)
#     return tf.keras.models.Model(inputs=IN, outputs=x)

#%%
######## Plot utils ##########

# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec

# def plot_loss_curves(df, prefix = 'rnx', save_path = '.'):
    
#     legendsize = 27
#     ticksize = 20
#     # Create a 2x2 grid for subplots using GridSpec
#     gs = GridSpec(3, 1, hspace=0.5)
    
#     # Create a new figure and get the axes objects for each subplot
#     fig = plt.figure(figsize=(20, 20))
    
#     # First subplot (top-left)
#     ax00 = fig.add_subplot(gs[0, 0])
#     ax01 = fig.add_subplot(gs[1, 0])
#     ax02 = fig.add_subplot(gs[2, 0])

#     ################################### SRDRN-MSE ###################################
#     ax = ax00
    
#     ax.set_facecolor('#F0F0F0') 
#     # ax.grid(True)
#     ax.plot(df['epoch'], df['loss'], linewidth=4, color='blue')
#     ax.plot(df['epoch'], df['val_loss'], linewidth=4, color='magenta')
    
#     # Find the index of the minimum val_loss
#     min_val_loss_idx = df['val_loss'].idxmin()
#     min_val_loss_epoch = df.at[min_val_loss_idx, 'epoch']
#     min_val_loss_value = df.at[min_val_loss_idx, 'val_loss']
    
#     # Annotate the lowest val_loss point with a star marker
#     ax.scatter(min_val_loss_epoch, min_val_loss_value, color='green', s=1000, marker='*', zorder=5)
    
#     # Set tick parameters
#     ax.tick_params(axis='both', which='major', labelsize=ticksize)
#     ax.set_xlabel('EPOCHS', fontsize = 25)
#     ax.set_ylabel('LOSS', fontsize = 25)
#     ax.set_title('LOSS', fontsize = 30, fontweight='bold')
    
#     ax.legend(['Train Loss', 'Validation Loss', f'Best Model (e{min_val_loss_idx})'], 
#               ncol=3, loc = 'upper center', fontsize = legendsize)
#     # ax.set_ylim(0, 0.2)
    
#     ################################### SRDRN-MAE ###################################
#     ax = ax01
    
#     ax.set_facecolor('#F0F0F0') 
#     # ax.grid(True)
#     ax.plot(df['epoch'], df['mean_absolute_error'], linewidth=4, color='blue')
#     ax.plot(df['epoch'], df['val_mean_absolute_error'], linewidth=4, color='magenta')
    
#     # Set tick parameters
#     ax.tick_params(axis='both', which='major', labelsize=ticksize)
#     ax.set_xlabel('EPOCHS', fontsize = 25)
#     ax.set_ylabel('MAE', fontsize = 25)
#     ax.set_title('MEAN ABSOLUTE ERROR', fontsize = 30, fontweight='bold')
#     # ax.set_ylim(0, 0.2)
    
#     ################################### SRDRN-WMAE ###################################
#     ax = ax02
    
#     ax.set_facecolor('#F0F0F0') 
#     # ax.grid(True)
#     ax.plot(df['epoch'], df['mean_squared_error'], linewidth=4, color='blue')
#     ax.plot(df['epoch'], df['val_mean_squared_error'], linewidth=4, color='magenta')
    
#     # Set tick parameters
#     ax.tick_params(axis='both', which='major', labelsize=ticksize)
#     ax.set_xlabel('EPOCHS', fontsize = 25)
#     ax.set_ylabel('MSE', fontsize = 25)
#     ax.set_title('MEAN SQUARED ERROR', fontsize = 30, fontweight='bold')
#     # ax.set_ylim(0, 0.2)
    
#     plt.savefig(f'{save_path}/{prefix}_traincurves.jpg', format='jpg', dpi=500, bbox_inches='tight', facecolor='w', edgecolor='w')
    
    
#%%

# import pandas as pd
# path = '/home/midhunm/AI4KLIM/EXPMNTS/P04A_PhySRDRN_Downscaling/RAWRUNS/P04A_200-210'

# for exp_id in range(200, 211):
#     plot_loss_curves(df = pd.read_csv(f'{path}/p04a_{exp_id}_logs.csv'), 
#                      prefix = f'p04a_{exp_id}', 
#                      save_path = path)
