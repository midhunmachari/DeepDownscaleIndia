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
from scipy.stats import pearsonr, ttest_ind
import hydroeval as he
from tqdm import tqdm
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Define evaluation metrics functions
def kge_calc(y_true, y_pred):
    evaluations = y_true[~np.isnan(y_true)].flatten()
    simulations = y_pred[~np.isnan(y_pred)].flatten()    
    
    kge, r, alpha, beta = he.evaluator(he.kgeprime, simulations, evaluations)
    return kge[0], r[0], alpha[0], beta[0]

def nse_calc(y_true, y_pred):
    evaluations = y_true[~np.isnan(y_true)].flatten()
    simulations = y_pred[~np.isnan(y_pred)].flatten()    
    
    nse = he.evaluator(he.nse, simulations, evaluations)
    return nse[0]

def rmse_calc(y_true, y_pred):
    evaluations = y_true[~np.isnan(y_true)].flatten()
    simulations = y_pred[~np.isnan(y_pred)].flatten()    
    
    rmse = he.evaluator(he.rmse, simulations, evaluations)
    return rmse[0]

def mare_calc(y_true, y_pred):    
    evaluations = y_true[~np.isnan(y_true)].flatten()
    simulations = y_pred[~np.isnan(y_pred)].flatten() 
    
    mare = he.evaluator(he.mare, simulations, evaluations)
    return mare[0]

def pbias_calc(y_true, y_pred):
    evaluations = y_true[~np.isnan(y_true)].flatten()
    simulations = y_pred[~np.isnan(y_pred)].flatten()   
    
    pbias = he.evaluator(he.pbias, simulations, evaluations)
    return pbias[0] * -1 # Reverse sign

def pcc_calc(y_true, y_pred):
    y_true = y_true[~np.isnan(y_true)].flatten()
    y_pred = y_pred[~np.isnan(y_pred)].flatten()
    
    pcc, pcc_pval = pearsonr(y_true, y_pred)
    return pcc, pcc_pval

def acc_calc(y_true, y_pred):
    """
    Calculate the Anomaly Correlation Coefficient (ACC).
    Anomalies for both y_true and y_pred are calculated w.r.t. the mean of y_true.
    """
    y_true = y_true[~np.isnan(y_true)].flatten()
    y_pred = y_pred[~np.isnan(y_pred)].flatten()
    print(f'\tMean of y_true and y_pred: {np.mean(y_true)} and {np.mean(y_pred)}')
    y_true_anomaly = y_true - np.mean(y_true)
    y_pred_anomaly = y_pred - np.mean(y_pred) 
    acc, acc_pval = pearsonr(y_true_anomaly, y_pred_anomaly)
    return acc, acc_pval

def ttest_calc(y_true, y_pred):
    """
    Calculate the Students T-Test independent.
    """
    y_true = y_true[~np.isnan(y_true)].flatten()
    y_pred = y_pred[~np.isnan(y_pred)].flatten()
    
    tstat, tstat_pval = ttest_ind(y_true, y_pred)
    return tstat, tstat_pval

def mbe_calc(y_true, y_pred):
    """Calculate Mean Bias Error (MBE)."""
    y_true = y_true[~np.isnan(y_true)].flatten()
    y_pred = y_pred[~np.isnan(y_pred)].flatten()
    
    mbe = np.mean(y_pred - y_true)
    return mbe


def tss_calc(y_true, y_pred, r_max = 0.999):
    """Calculate Taylor Skill Score (TSS).""" 
    y_true = y_true[~np.isnan(y_true)].flatten()
    y_pred = y_pred[~np.isnan(y_pred)].flatten()
    
    r = pearsonr(y_true, y_pred)[0]
    y_true_std = np.std(y_true)
    y_pred_std = np.std(y_pred)
    numer = 4 * ((1 + r) ** 2)
    denom = (((y_pred_std / y_true_std) + (y_true_std / y_pred_std)) ** 2) * ((1 + r_max) ** 2)
    tss = numer / denom
    return tss

def evs_calc(y_true, y_pred):
    """Calculate Explained Variance Score (TSS).""" 
    y_true = y_true[~np.isnan(y_true)].flatten()
    y_pred = y_pred[~np.isnan(y_pred)].flatten()
    
    evs = explained_variance_score(y_true, y_pred)
    return evs
       
def r2_calc(y_true, y_pred):
    """Calculate R2 Score (R2).""" 
    y_true = y_true[~np.isnan(y_true)].flatten()
    y_pred = y_pred[~np.isnan(y_pred)].flatten()
    
    r2 = r2_score(y_true, y_pred)
    return r2

def mse_calc(y_true, y_pred):
    """Calculate Mean Squared Error (MSE).""" 
    y_true = y_true[~np.isnan(y_true)].flatten()
    y_pred = y_pred[~np.isnan(y_pred)].flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    return mse

def mae_calc(y_true, y_pred):
    """Calculate Mean Absolute Error (MAE).""" 
    y_true = y_true[~np.isnan(y_true)].flatten()
    y_pred = y_pred[~np.isnan(y_pred)].flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    return mae

def mape_calc(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE).""" 
    y_true = y_true[~np.isnan(y_true)].flatten()
    y_pred = y_pred[~np.isnan(y_pred)].flatten()
    
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mape

def nrmse_calc(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE).""" 
    y_true = y_true[~np.isnan(y_true)].flatten()
    y_pred = y_pred[~np.isnan(y_pred)].flatten()

    nrmse = rmse_calc(y_true, y_pred) / (np.max(y_true) - np.min(y_true))
    return nrmse
    
#%% 

def make_eval_table_generic(data_dict, suffix='x.x.x', csv_save_path = '.'):
    
    results = []
    
    y_true = xr.open_dataset(data_dict['REF'][0]).prec.data.squeeze()

    for exp_id, (exp_data_path, label) in data_dict.items():

        if exp_id=='REF':
            continue
        
        else:
        
            print(f'\nCalculating metrics for ... {exp_id}: {label}')
            print(f'Loading ... {exp_data_path}')
            
            y_pred = xr.open_dataset(exp_data_path).prec.data.squeeze()
            
            metrics_values = {
                                
                                'KGE'       : kge_calc(y_true, y_pred)[0],
                                'R'         : kge_calc(y_true, y_pred)[1],
                                'ALPHA'     : kge_calc(y_true, y_pred)[2],
                                'BETA'      : kge_calc(y_true, y_pred)[3],
                                'TSS'       : tss_calc(y_true, y_pred),
                                'NSE'       : nse_calc(y_true, y_pred),
                                'RMSE'      : rmse_calc(y_true, y_pred),
                                'NRMSE'     : nrmse_calc(y_true, y_pred),
                                'MARE'      : mare_calc(y_true, y_pred),
                                'PBIAS'     : pbias_calc(y_true, y_pred),
                                'EVS'       : evs_calc(y_true, y_pred),
                                'R2'        : r2_calc(y_true, y_pred),
                                'MSE'       : mse_calc(y_true, y_pred),
                                'MAE'       : mae_calc(y_true, y_pred),
                                'MAPE'      : mape_calc(y_true, y_pred),
                                'MBE'       : mbe_calc(y_true, y_pred),
                                'PCC'       : pcc_calc(y_true, y_pred)[0],
                                'PCC_PVAL'  : pcc_calc(y_true, y_pred)[1],
                                'ACC'       : acc_calc(y_true, y_pred)[0],
                                'ACC_PVAL'  : acc_calc(y_true, y_pred)[1], 
                                'TSTAT'     : ttest_calc(y_true, y_pred)[0],
                                'TSTAT_PVAL': ttest_calc(y_true, y_pred)[1],
                                'MAX_VAL'   : np.nanmax(y_pred)
                            }
        
            # Round metrics to the desired decimal place
            rd = 3
            rounded_metrics_values = {key: round(value, rd) for key, value in metrics_values.items()}

            kge_value = metrics_values['KGE']
            print(f'\tKGE: {kge_value}')
            
            # Append results to the list
            results.append([exp_id, label] + list(rounded_metrics_values.values()))
            
        # Give headers to CSV file
        header = ['EXP_ID', 'LABEL'] + list(rounded_metrics_values.keys())
        
        #################### SAVE RESULTS ####################
        
        csv_name = f'TABLE_EVALMETRICS_{suffix}.csv'
        
        
        # Convert the results to a DataFrame
        df = pd.DataFrame(results, columns=header)
        
        print("\nFinal Results:")
        print(df.to_string(index=False))
        
        df.to_csv(f'{csv_save_path}/{csv_name}', index=False)
        
        print(f'\nResults saved at {csv_save_path}/{csv_name}')

def filter_months(dataarray, filter_months_list = [6, 7, 8, 9]):
    """Filter months from an xarray dataarray"""
    return dataarray.sel(time=dataarray['time.month'].isin(filter_months_list))

def mask_array(dataarray, maskarray):
    """Mask an xarray dataarray"""
    if isinstance(maskarray, xr.DataArray):
        maskarray = maskarray.values
    if dataarray.shape[1:2] != maskarray.shape[1:2]:
        raise ValueError("dataarray and maskarray must have the same shape")
    return dataarray*maskarray

def make_eval_table(data_dict,
                   mask_dict=None, 
                   jjas_only=False, 
                   suffix='OK', 
                   csv_save_path = '.'):

    if mask_dict is not None:
        
        results = []
        
        for zone, (zone_mask_path, zone_name) in mask_dict.items():
    
            print(f'\nCalculating metrics for ... {zone}: {zone_name}')
            print(f'Loading ... {zone_mask_path}')
            
            mask = xr.open_dataset(zone_mask_path).mask
            
            y_true = xr.open_dataset(data_dict['REF'][0]).prec
            if jjas_only:
                y_true = filter_months(y_true, filter_months_list = [6, 7, 8, 9])
            y_true = mask_array(y_true, mask)
            y_true = y_true.data.squeeze()
            
            for exp_id, (exp_data_path, label) in data_dict.items():
            
                if exp_id=='REF':
                    continue
            
                else:
                    print(f'\n\tCalculating metrics for ... {exp_id}: {label}')
                    print(f'\tLoading ... {exp_data_path}')
                    
                    y_pred = xr.open_dataset(exp_data_path).prec
                    if jjas_only:
                        y_pred = filter_months(y_pred, filter_months_list = [6, 7, 8, 9])
                    y_pred = mask_array(y_pred, mask)
                    y_pred = y_pred.data.squeeze()
                    
                    metrics_values = {
                                
                                'KGE'       : kge_calc(y_true, y_pred)[0],
                                'R'         : kge_calc(y_true, y_pred)[1],
                                'ALPHA'     : kge_calc(y_true, y_pred)[2],
                                'BETA'      : kge_calc(y_true, y_pred)[3],
                                'TSS'       : tss_calc(y_true, y_pred),
                                'NSE'       : nse_calc(y_true, y_pred),
                                'RMSE'      : rmse_calc(y_true, y_pred),
                                'NRMSE'     : nrmse_calc(y_true, y_pred),
                                'MARE'      : mare_calc(y_true, y_pred),
                                'PBIAS'     : pbias_calc(y_true, y_pred),
                                'EVS'       : evs_calc(y_true, y_pred),
                                'R2'        : r2_calc(y_true, y_pred),
                                'MSE'       : mse_calc(y_true, y_pred),
                                'MAE'       : mae_calc(y_true, y_pred),
                                'MAPE'      : mape_calc(y_true, y_pred),
                                'MBE'       : mbe_calc(y_true, y_pred),
                                'PCC'       : pcc_calc(y_true, y_pred)[0],
                                'PCC_PVAL'  : pcc_calc(y_true, y_pred)[1],
                                'ACC'       : acc_calc(y_true, y_pred)[0],
                                'ACC_PVAL'  : acc_calc(y_true, y_pred)[1], 
                                'TSTAT'     : ttest_calc(y_true, y_pred)[0],
                                'TSTAT_PVAL': ttest_calc(y_true, y_pred)[1],
                                'MAX_VAL'   : np.nanmax(y_pred)
                            }
        
                    # Round metrics to the desired decimal place
                    rounded_metrics_values = {key: round(value, 8) for key, value in metrics_values.items()}
    
                    nse_value = metrics_values['NSE']
                    print(f'\tNSE: {nse_value}')
                    
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

    else:
        make_eval_table_generic(data_dict, suffix=suffix, csv_save_path = csv_save_path)

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

    'C01': [f'{DATA_PATH}/quantile_mapping_disagregated_out.nc' , 'BCSD'],

    }

MASK_PATH = "/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/PLTDATA/MASKFILES"
mask_dict = {
    'IND': [f'{MASK_PATH}/India.nc'             , 'INDIA'], 
    'MCZ': [f'{MASK_PATH}/Monsoon_Core_Zone.nc' , 'MONSOON CORE ZONE'], 
    'WG' : [f'{MASK_PATH}/Western_Ghats.nc'     , 'WESTERN GHATS'], 
    'SP' : [f'{MASK_PATH}/South_Peninsular.nc'  , 'SOUTHERN PENINSULA'], 
    'CNE': [f'{MASK_PATH}/Central_Northeast.nc' , 'CENTRAL NORTHEAST'], 
    'NE' : [f'{MASK_PATH}/Northeast.nc'         , 'NORTHEAST'], 
    'WC' : [f'{MASK_PATH}/West_Central.nc'      , 'WEST CENTRAL'], 
    'NW' : [f'{MASK_PATH}/Northwest.nc'         , 'NORTHWEST'], 
    'HIM': [f'{MASK_PATH}/Himalaya.nc'          , 'THE HIMALAYAS'], 
    'HR' : [f'{MASK_PATH}/Hilly_Regions.nc'     , 'HILLY REGION'],
 }

SAVE_PATH = "/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/PLTDATA"

if __name__ == "__main__":
    make_eval_table(data_dict, mask_dict, jjas_only=False, suffix='ALLD_v1.0.0', csv_save_path = SAVE_PATH)
    make_eval_table(data_dict, mask_dict, jjas_only=True,  suffix='JJAS_v1.0.0', csv_save_path = SAVE_PATH)


