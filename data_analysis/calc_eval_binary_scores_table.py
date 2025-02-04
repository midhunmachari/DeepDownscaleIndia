#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:18:22 2024

@author: midhunm
"""

import os
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


    
#%% 
import numpy as np
import xarray as xr

def calculate_binary_metrics(obs_events, pred_events):
    """
    Calculate binary classification metrics: CSI, POD, FAR, and ETS.

    Parameters:
    - obs_events: array-like
        Array of observed event values (binary).
    - pred_events: array-like
        Array of predicted event values (binary).

    Returns:
    - CSI: Critical Success Index
    - POD: Probability of Detection
    - FAR: False Alarm Rate
    - ETS: Equitable Threat Score
    """
    # Calculate confusion matrix elements
    TP = np.sum((obs_events == 1) & (pred_events == 1))
    FP = np.sum((obs_events == 0) & (pred_events == 1))
    FN = np.sum((obs_events == 1) & (pred_events == 0))
    TN = np.sum((obs_events == 0) & (pred_events == 0))

    # Compute metrics
    CSI = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    POD = TP / (TP + FN) if (TP + FN) > 0 else 0
    FAR = FP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Calculate ETS
    total = TP + FP + FN + TN
    random_hits = ((TP + FN) * (TP + FP)) / total if total > 0 else 0  # Expected hits by chance
    ETS = (TP - random_hits) / (TP + FP + FN - random_hits) if (TP + FP + FN - random_hits) > 0 else 0

    return CSI, POD, FAR, ETS, TP, FP, FN, TN 

def convert_percentile_events_to_binary(ds, ref_ds=None, thresh_ds=None, percentile=99):
    """
    Converts values in a dataset to binary based on a threshold derived from a reference dataset
    or a predefined threshold dataset.
    
    Parameters:
    - ds: xarray.Dataset or xarray.DataArray
        The dataset or data array to be converted to binary.
    - ref_ds: xarray.Dataset or xarray.DataArray, optional
        Reference dataset to calculate the threshold from, if `thresh_ds` is not provided.
    - thresh_ds: xarray.DataArray, optional
        Predefined threshold dataset to use for conversion. Overrides `ref_ds` and `percentile`.
    - percentile: float, optional
        The percentile value to use when calculating the threshold from `ref_ds`. Defaults to 99.

    Returns:
    - binary_ds: xarray.DataArray
        Binary dataset or data array where values above or equal to the threshold are 1, otherwise 0.
    """
    
    # Determine the threshold based on precedence
    if thresh_ds is not None:
        threshold = thresh_ds
    elif ref_ds is not None:
        threshold = ref_ds.quantile(percentile / 100.0, dim='time')
    else:
        raise ValueError("Either `ref_ds` or `thresh_ds` must be provided to determine the threshold.")
    
    # Convert values to binary: 1 for values above or equal to the threshold, 0 otherwise
    binary_ds = xr.where(ds >= threshold, 1, 0)
    
    return binary_ds

def convert_percentile_events_to_binary_pair(ds_true, ds_pred, thresh_ds=None, percentile=99):
    """
    Converts values in a dataset to binary based on a threshold.

    Parameters:
    - ds: xarray.DataArray
        Input data to convert to binary.
    - ref_ds: xarray.DataArray, optional
        Reference dataset to calculate the threshold.
    - thresh_ds: xarray.DataArray, optional
        Provided threshold to use directly.
    - percentile: int, optional
        Percentile to use for threshold calculation from `ref_ds` if `thresh_ds` is not provided.

    Returns:
    - binary_ds: xarray.DataArray
        Binary dataset where values are 1 if above threshold, 0 otherwise.
    """
    # Determine the threshold based on precedence
    if thresh_ds is not None:
        print('Using provided threshold dataset...')
        threshold = thresh_ds
    else:
        print(f'Calculating threshold using the {percentile}th percentile of reference data...')
        threshold = ds_true.quantile(percentile / 100.0, dim='time')
    
    # Convert values to binary: 1 for values above or equal to the threshold, 0 otherwise
    true_binary_ds = xr.where(ds_true >= threshold, 1, 0)
    pred_binary_ds = xr.where(ds_pred >= threshold, 1, 0)
    
    return true_binary_ds, pred_binary_ds

def convert_threshold_events_to_binary(ds, threshold=1):
    """
    Converts values in a dataset to binary based on a threshold.
    """
    
    # Convert values to binary: 1 for values above or equal to the threshold, 0 otherwise
    binary_ds = xr.where(ds >= threshold, 1, 0)

    return binary_ds

def convert_threshold_events_to_binary_pair(ds_true, ds_pred, threshold=1):
    """
    Converts values in a dataset to binary based on a threshold.
    """
    
    # Convert values to binary: 1 for values above or equal to the threshold, 0 otherwise
    true_binary_ds = xr.where(ds_true >= threshold, 1, 0)
    pred_binary_ds = xr.where(ds_pred >= threshold, 1, 0)
    
    return true_binary_ds, pred_binary_ds


def mask_array(dataarray, maskarray):
    """
    Masks an xarray.DataArray using a provided mask array.

    Parameters:
    - dataarray: xarray.DataArray
        Data to apply the mask to.
    - maskarray: xarray.DataArray or numpy.ndarray
        The mask array. Values of 1 will keep the data, and values of 0 will mask them.

    Returns:
    - masked_data: xarray.DataArray
        Masked data array.
    """
    if isinstance(maskarray, xr.DataArray):
        maskarray = maskarray.values

    if dataarray.shape[1:3] != maskarray.shape[1:3]:
        raise ValueError("Data array and mask array must have the same shape in the latitude and longitude dimensions.")
    
    return dataarray * maskarray


def calculate_binary_scores(true_binary_arr, pred_binary_arr):

    """
    Calculates binary evaluation metrics (CSI, POD, FAR, ETS) between observed and predicted binary datasets.

    """
    observed = true_binary_arr.flatten()
    predicted = pred_binary_arr.flatten()

    # Remove NaN values
    valid_mask = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid_mask]
    predicted = predicted[valid_mask]

    # Ensure observed and predicted have the same shape after NaN removal
    assert observed.shape == predicted.shape, "Observed and predicted arrays must have the same shape after NaN removal."
    print(f"\tObserved shape: {observed.shape}\n\tPredicted shape: {predicted.shape}")

    # Calculate binary metrics
    CSI, POD, FAR, ETS, TP, FP, FN, TN = calculate_binary_metrics(observed, predicted)

    # Prepare and return the result dictionary
    return {
        "POD": POD,
        "FAR": FAR,
        "CSI": CSI,
        "ETS": ETS,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
    }

def make_percentile_events_scores_table(data_dict,
                                        percentile_thresh=99,
                                        mask_dict=None, 
                                        suffix='OK', 
                                        csv_save_path = '.'
                                        ):

    if mask_dict is not None:
        
        results = []
        
        for zone, (zone_mask_path, zone_name) in mask_dict.items():
    
            print(f'\nCalculating metrics for ... {zone}: {zone_name}')
            print(f'Loading ... {zone_mask_path}')
            
            mask = xr.open_dataset(zone_mask_path).mask
            y_true = y_true_ref = xr.open_dataset(data_dict['REF'][0])

            y_true = convert_percentile_events_to_binary(y_true, ref_ds=y_true, thresh_ds=None, percentile=percentile_thresh)
            y_true = mask_array(y_true['prec'], mask)
            y_true = y_true.data.squeeze()
            
            for exp_id, (exp_data_path, label) in data_dict.items():
            
                if exp_id=='REF':
                    continue
            
                else:
                    print(f'\n\tCalculating metrics for ... {exp_id}: {label}')
                    print(f'\tLoading ... {exp_data_path}')
                    
                    y_pred = xr.open_dataset(exp_data_path)
                    y_pred = convert_percentile_events_to_binary(y_pred, ref_ds=y_true_ref, thresh_ds=None, percentile=percentile_thresh)
                    y_pred = mask_array(y_pred['prec'], mask)
                    y_pred = y_pred.data.squeeze()

                    
                    metrics_values = calculate_binary_scores(y_true, y_pred)
        
                    # Round metrics to the desired decimal place
                    rounded_metrics_values = {key: round(value, 8) for key, value in metrics_values.items()}
    
                    value = metrics_values['ETS']
                    print(f'\tETS: {value}')
                    
                    # Append results to the list
                    results.append([zone, zone_name, exp_id, label] + list(rounded_metrics_values.values()))
            
        # Give headers to CSV file
        header = ['ZONE', 'ZONE_NAME', 'EXP_ID', 'LABEL'] + list(rounded_metrics_values.keys())

        #################### SAVE RESULTS ####################
    
        # Convert the results to a DataFrame
        df = pd.DataFrame(results, columns=header)
        
        print("\nFinal Results:")
        print(df.to_string(index=False))

        csv_name = f'TABLE_EVALMETRICS_{suffix}.csv'
        df.to_csv(f'{csv_save_path}/{csv_name}', index=False)
        
        print(f'\nResults saved at {csv_save_path}/{csv_name}')

    # else:
    #     make_eval_table_generic(data_dict, suffix=suffix, csv_save_path = csv_save_path)
        
def make_threshold_events_scores_table(data_dict,
                                        threshold=1,
                                        mask_dict=None, 
                                        suffix='OK', 
                                        csv_save_path = '.'
                                        ):

    if mask_dict is not None:
        
        results = []
        
        for zone, (zone_mask_path, zone_name) in mask_dict.items():
    
            print(f'\nCalculating metrics for ... {zone}: {zone_name}')
            print(f'Loading ... {zone_mask_path}')
            
            mask = xr.open_dataset(zone_mask_path).mask
            y_true = y_true_ref = xr.open_dataset(data_dict['REF'][0])

            y_true = convert_threshold_events_to_binary(y_true, threshold)
            y_true = mask_array(y_true['prec'], mask)
            y_true = y_true.data.squeeze()
            
            for exp_id, (exp_data_path, label) in data_dict.items():
            
                if exp_id=='REF':
                    continue
            
                else:
                    print(f'\n\tCalculating metrics for ... {exp_id}: {label}')
                    print(f'\tLoading ... {exp_data_path}')
                    
                    y_pred = xr.open_dataset(exp_data_path)
                    y_pred = convert_threshold_events_to_binary(y_pred, threshold)
                    y_pred = mask_array(y_pred['prec'], mask)
                    y_pred = y_pred.data.squeeze()

                    
                    metrics_values = calculate_binary_scores(y_true, y_pred)
        
                    # Round metrics to the desired decimal place
                    rounded_metrics_values = {key: round(value, 8) for key, value in metrics_values.items()}
    
                    value = metrics_values['ETS']
                    print(f'\tETS: {value}')
                    
                    # Append results to the list
                    results.append([zone, zone_name, exp_id, label] + list(rounded_metrics_values.values()))
            
        # Give headers to CSV file
        header = ['ZONE', 'ZONE_NAME', 'EXP_ID', 'LABEL'] + list(rounded_metrics_values.keys())

        #################### SAVE RESULTS ####################
    
        # Convert the results to a DataFrame
        df = pd.DataFrame(results, columns=header)
        
        print("\nFinal Results:")
        print(df.to_string(index=False))

        csv_name = f'TABLE_EVALMETRICS_{suffix}.csv'
        df.to_csv(f'{csv_save_path}/{csv_name}', index=False)
        
        print(f'\nResults saved at {csv_save_path}/{csv_name}')

    # else:
    #     make_eval_table_generic(data_dict, suffix=suffix, csv_save_path = csv_save_path)

#%% Calculte and save csv.


DATA_PATH = "/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/GENDATA"
data_dict = {  
    
    'REF': [f'{DATA_PATH}/ref_imda_012_data.nc', 'IMDAA'],
    
    'N01_L01_I01': [f'{DATA_PATH}/p07a_n01_l01_i01_b01_r01_012_out.nc', 'DENSE-WMAE-i01'  ],
    'N01_L01_I02': [f'{DATA_PATH}/p07a_n01_l01_i02_b01_r01_012_out.nc', 'DENSE-WMAE-i02'  ],
    'N01_L01_I03': [f'{DATA_PATH}/p07a_n01_l01_i03_b01_r01_012_out.nc', 'DENSE-WMAE-i03'  ],
    'N01_L02_I01': [f'{DATA_PATH}/p07a_n01_l02_i01_b01_r01_012_out.nc', 'DENSE-GAMMA-i01' ],
    'N01_L02_I02': [f'{DATA_PATH}/p07a_n01_l02_i02_b01_r01_012_out.nc', 'DENSE-GAMMA-i02' ],
    'N01_L02_I03': [f'{DATA_PATH}/p07a_n01_l02_i03_b01_r01_012_out.nc', 'DENSE-GAMMA-i03' ],

    'N02_L01_I01': [f'{DATA_PATH}/p07a_n02_l01_i01_b01_r01_012_out.nc', 'CNN-WMAE-i01'    ],
    'N02_L01_I02': [f'{DATA_PATH}/p07a_n02_l01_i02_b01_r01_012_out.nc', 'CNN-WMAE-i02'    ],
    'N02_L01_I03': [f'{DATA_PATH}/p07a_n02_l01_i03_b01_r01_012_out.nc', 'CNN-WMAE-i03'    ],
    'N02_L02_I01': [f'{DATA_PATH}/p07a_n02_l02_i01_b01_r01_012_out.nc', 'CNN-GAMMA-i01'   ],
    'N02_L02_I02': [f'{DATA_PATH}/p07a_n02_l02_i02_b01_r01_012_out.nc', 'CNN-GAMMA-i02'   ],
    'N02_L02_I03': [f'{DATA_PATH}/p07a_n02_l02_i03_b01_r01_012_out.nc', 'CNN-GAMMA-i03'   ],
    
    'N03_L01_I01': [f'{DATA_PATH}/p07a_n03_l01_i01_b01_r01_012_out.nc', 'FSRCNN-WMAE-i01' ],
    'N03_L01_I02': [f'{DATA_PATH}/p07a_n03_l01_i02_b01_r01_012_out.nc', 'FSRCNN-WMAE-i02' ],
    'N03_L01_I03': [f'{DATA_PATH}/p07a_n03_l01_i03_b01_r01_012_out.nc', 'FSRCNN-WMAE-i03' ],
    'N03_L02_I01': [f'{DATA_PATH}/p07a_n03_l02_i01_b01_r01_012_out.nc', 'FSRCNN-GAMMA-i01'],
    'N03_L02_I02': [f'{DATA_PATH}/p07a_n03_l02_i02_b01_r01_012_out.nc', 'FSRCNN-GAMMA-i02'],
    'N03_L02_I03': [f'{DATA_PATH}/p07a_n03_l02_i03_b01_r01_012_out.nc', 'FSRCNN-GAMMA-i03'],

    'S01_L01_I01': [f'{DATA_PATH}/p07a_s01_l01_i01_b01_r01_012_out.nc', 'SRDRN1-WMAE-i01' ],
    'S01_L01_I02': [f'{DATA_PATH}/p07a_s01_l01_i02_b01_r01_012_out.nc', 'SRDRN1-WMAE-i02' ],
    'S01_L01_I03': [f'{DATA_PATH}/p07a_s01_l01_i03_b01_r01_012_out.nc', 'SRDRN1-WMAE-i03' ],
    'S01_L02_I01': [f'{DATA_PATH}/p07a_s01_l02_i01_b01_r01_012_out.nc', 'SRDRN1-GAMMA-i01'],
    'S01_L02_I02': [f'{DATA_PATH}/p07a_s01_l02_i02_b01_r01_012_out.nc', 'SRDRN1-GAMMA-i02'],
    'S01_L02_I03': [f'{DATA_PATH}/p07a_s01_l02_i03_b01_r01_012_out.nc', 'SRDRN1-GAMMA-i03'],
    
    'S02_L01_I01': [f'{DATA_PATH}/p07a_s02_l01_i01_b01_r01_012_out.nc', 'SRDRN2-WMAE-i01' ],
    'S02_L01_I02': [f'{DATA_PATH}/p07a_s02_l01_i02_b01_r01_012_out.nc', 'SRDRN2-WMAE-i02' ],
    'S02_L01_I03': [f'{DATA_PATH}/p07a_s02_l01_i03_b01_r01_012_out.nc', 'SRDRN2-WMAE-i03' ],
    'S02_L02_I01': [f'{DATA_PATH}/p07a_s02_l02_i01_b01_r01_012_out.nc', 'SRDRN2-GAMMA-i01'],
    'S02_L02_I02': [f'{DATA_PATH}/p07a_s02_l02_i02_b01_r01_012_out.nc', 'SRDRN2-GAMMA-i02'],
    'S02_L02_I03': [f'{DATA_PATH}/p07a_s02_l02_i03_b01_r01_012_out.nc', 'SRDRN2-GAMMA-i03'],

    'S03_L01_I01': [f'{DATA_PATH}/p07a_s03_l01_i01_b01_r01_012_out.nc', 'SRDRN3-WMAE-i01' ],
    'S03_L01_I02': [f'{DATA_PATH}/p07a_s03_l01_i02_b01_r01_012_out.nc', 'SRDRN3-WMAE-i02' ],
    'S03_L01_I03': [f'{DATA_PATH}/p07a_s03_l01_i03_b01_r01_012_out.nc', 'SRDRN3-WMAE-i03' ],
    'S03_L02_I01': [f'{DATA_PATH}/p07a_s03_l02_i01_b01_r01_012_out.nc', 'SRDRN3-GAMMA-i01'],
    'S03_L02_I02': [f'{DATA_PATH}/p07a_s03_l02_i02_b01_r01_012_out.nc', 'SRDRN3-GAMMA-i02'],
    'S03_L02_I03': [f'{DATA_PATH}/p07a_s03_l02_i03_b01_r01_012_out.nc', 'SRDRN3-GAMMA-i03'],

    'U01_L01_I01': [f'{DATA_PATH}/p07a_u01_l01_i01_b01_r01_012_out.nc', 'UNET-WMAE-i01'   ],
    'U01_L01_I02': [f'{DATA_PATH}/p07a_u01_l01_i02_b01_r01_012_out.nc', 'UNET-WMAE-i02'   ],
    'U01_L01_I03': [f'{DATA_PATH}/p07a_u01_l01_i03_b01_r01_012_out.nc', 'UNET-WMAE-i03'   ],
    'U01_L02_I01': [f'{DATA_PATH}/p07a_u01_l02_i01_b01_r01_012_out.nc', 'UNET-GAMMA-i01'  ],
    'U01_L02_I02': [f'{DATA_PATH}/p07a_u01_l02_i02_b01_r01_012_out.nc', 'UNET-GAMMA-i02'  ],
    'U01_L02_I03': [f'{DATA_PATH}/p07a_u01_l02_i03_b01_r01_012_out.nc', 'UNET-GAMMA-i03'  ],

    'U02_L01_I01': [f'{DATA_PATH}/p07a_u02_l01_i01_b01_r01_012_out.nc', 'XNET-WMAE-i01'   ],
    'U02_L01_I02': [f'{DATA_PATH}/p07a_u02_l01_i02_b01_r01_012_out.nc', 'XNET-WMAE-i02'   ],
    'U02_L01_I03': [f'{DATA_PATH}/p07a_u02_l01_i03_b01_r01_012_out.nc', 'XNET-WMAE-i03'   ],
    'U02_L02_I01': [f'{DATA_PATH}/p07a_u02_l02_i01_b01_r01_012_out.nc', 'XNET-GAMMA-i01'  ],
    'U02_L02_I02': [f'{DATA_PATH}/p07a_u02_l02_i02_b01_r01_012_out.nc', 'XNET-GAMMA-i02'  ],
    'U02_L02_I03': [f'{DATA_PATH}/p07a_u02_l02_i03_b01_r01_012_out.nc', 'XNET-GAMMA-i03'  ],

    'C01': [f'{DATA_PATH}/c01_quantile_mapping_disagregated_out.nc' , 'BCSD'],

    }

MASK_PATH = "/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/PLTDATA/MASKFILES"
mask_dict = {
    'IND': [f'{MASK_PATH}/India.nc'             , 'INDIA'], 
    # 'MCZ': [f'{MASK_PATH}/Monsoon_Core_Zone.nc' , 'MONSOON CORE ZONE'], 
    # 'WG' : [f'{MASK_PATH}/Western_Ghats.nc'     , 'WESTERN GHATS'], 
    # 'SP' : [f'{MASK_PATH}/South_Peninsular.nc'  , 'SOUTHERN PENINSULA'], 
    # 'CNE': [f'{MASK_PATH}/Central_Northeast.nc' , 'CENTRAL NORTHEAST'], 
    # 'NE' : [f'{MASK_PATH}/Northeast.nc'         , 'NORTHEAST'], 
    # 'WC' : [f'{MASK_PATH}/West_Central.nc'      , 'WEST CENTRAL'], 
    # 'NW' : [f'{MASK_PATH}/Northwest.nc'         , 'NORTHWEST'], 
    # 'HIM': [f'{MASK_PATH}/Himalaya.nc'          , 'THE HIMALAYAS'], 
    # 'HR' : [f'{MASK_PATH}/Hilly_Regions.nc'     , 'HILLY REGION'],
 }

SAVE_PATH = "/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/PLTDATA"

if __name__ == "__main__":
    # make_percentile_events_scores_table(data_dict,
    #                                     percentile_thresh=99,
    #                                     mask_dict=mask_dict, 
    #                                     suffix='BINARY_METRICS_IND_A99P_v1', 
    #                                     csv_save_path = SAVE_PATH
    #                                     )
    
    # make_percentile_events_scores_table(data_dict,
    #                                     percentile_thresh=95,
    #                                     mask_dict=mask_dict, 
    #                                     suffix='BINARY_METRICS_IND_A95P_v1', 
    #                                     csv_save_path = SAVE_PATH
                                        # )
    
    make_threshold_events_scores_table(data_dict,
                                        threshold=1,
                                        mask_dict=mask_dict, 
                                        suffix='BINARY_METRICS_IND_A1MM_v1', 
                                        csv_save_path = SAVE_PATH
                                        )


