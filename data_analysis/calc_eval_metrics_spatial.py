#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:18:22 2024

@author: midhunm
"""
import os
import pickle
import numpy as np
import xarray as xr
from scipy.stats import pearsonr, ttest_ind
import hydroeval as he
from tqdm import tqdm
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import multiprocessing as mp

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
    
# Define other utility functions

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def calculate_evalmetrics_spatial(exp_dict, data_path, save_path, varname='prec'):
    
    ref_data = exp_dict['REF'][0]
    print(ref_data)
    refer_data = xr.open_dataset(f'{data_path}/{ref_data}' if data_path is None else ref_data)
    refer_variable = refer_data[varname].load()

    # Create empty arrays for storage
    metrics_dict = {
        'KGE', 'R', 'ALPHA', 'BETA', 'TSS', 'NSE', 'RMSE', 'NRMSE', 'MARE', 'PBIAS', 
        'EVS', 'R2', 'MSE', 'MAE', 'MAPE', 'MBE', 
        'PCC', 'PCC_PVAL', 'ACC', 'ACC_PVAL',  'TSTAT', 'TSTAT_PVAL',
        }
    
    metrics_arrays = {metric: np.zeros_like(refer_variable.isel(time=0).values) for metric in metrics_dict}

    for exp_id, (exp_data, label) in exp_dict.items():

        print(f'Processing ... {exp_id}: {label}')

        if exp_id == 'REF':
            print('Skipping ...')
            continue

        model_data = xr.open_dataset(f'{data_path}/{exp_data}' if data_path is None else exp_data)
        model_variable = model_data[varname].load()

        # Iterate across all lon-lat points
        for lat_idx in tqdm(range(refer_variable.shape[1])):

            for lon_idx in range(refer_variable.shape[2]):
                    
                refer_series = refer_variable.isel(lat=lat_idx, lon=lon_idx).values.flatten()
                model_series = model_variable.isel(lat=lat_idx, lon=lon_idx).values.flatten() # Edit here

                if np.isnan(model_series).all() and np.isnan(refer_series).all():
                    metrics_values = {metric: np.nan for metric in metrics_dict}
                
                else:
                    metrics_values = {
                        
                        'KGE'       : kge_calc(refer_series, model_series)[0],
                        'R'         : kge_calc(refer_series, model_series)[1],
                        'ALPHA'     : kge_calc(refer_series, model_series)[2],
                        'BETA'      : kge_calc(refer_series, model_series)[3],
                        'TSS'       : tss_calc(refer_series, model_series),
                        'NSE'       : nse_calc(refer_series, model_series),
                        'RMSE'      : rmse_calc(refer_series, model_series),
                        'NRMSE'     : nrmse_calc(refer_series, model_series),
                        'MARE'      : mare_calc(refer_series, model_series),
                        'PBIAS'     : pbias_calc(refer_series, model_series),
                        'EVS'       : evs_calc(refer_series, model_series),
                        'R2'        : r2_calc(refer_series, model_series),
                        'MSE'       : mse_calc(refer_series, model_series),
                        'MAE'       : mae_calc(refer_series, model_series),
                        'MAPE'      : mape_calc(refer_series, model_series),
                        'MBE'       : mbe_calc(refer_series, model_series),
                        'PCC'       : pcc_calc(refer_series, model_series)[0],
                        'PCC_PVAL'  : pcc_calc(refer_series, model_series)[1],
                        'ACC'       : acc_calc(refer_series, model_series)[0],
                        'ACC_PVAL'  : acc_calc(refer_series, model_series)[1], 
                        'TSTAT'     : ttest_calc(refer_series, model_series)[0],
                        'TSTAT_PVAL': ttest_calc(refer_series, model_series)[1],

                    }

                # Assign values to arrays
                for metric, value in metrics_values.items():
                    metrics_arrays[metric][lat_idx, lon_idx] = value

        try:
            # Create xarray datasets for each metric
            datasets = {metric: xr.Dataset({metric: (['lat', 'lon'], values)},  # Edit here
                                        coords={'lat': model_variable['lat'],   # Edit here
                                                'lon': model_variable['lon']})  # Edit here
                        for metric, values in metrics_arrays.items()}

            # Combine datasets into a single dataset
            combined_data = xr.Dataset({f'{exp_id}_{metric.upper()}': dataset[metric] for metric, dataset in datasets.items()},
                                    coords={'lat': model_variable['lat'],       # Edit here
                                            'lon': model_variable['lon']        # Edit here
                                            })

            # Save the combined dataset to the same NetCDF file
            output_file_path = f'{save_path}/TEMP2MERGE_{exp_id}.nc'
        
            combined_data.to_netcdf(output_file_path, mode='w', format='NETCDF4')
            print(f'\nCreated {output_file_path}')
        
        except Exception as e:
            print(f"Failed to save NetCDF file: {e}")
            # Save as pickle file if NetCDF save fails
            pickle_output_file_path = f'{save_path}/TEMP2MERGE_{exp_id}.pkl'
            with open(pickle_output_file_path, 'wb') as f:
                pickle.dump(combined_data, f)
            print(f"Saved as pickle object: {pickle_output_file_path}")

def process_latitude(lat_idx, refer_variable, model_variable, metrics_dict):
    """Process a single latitude index."""
    metrics_values_list = []
    for lon_idx in range(refer_variable.shape[2]):
        refer_series = refer_variable.isel(lat=lat_idx, lon=lon_idx).values.flatten()
        model_series = model_variable.isel(lat=lat_idx, lon=lon_idx).values.flatten()

        if np.isnan(model_series).all() and np.isnan(refer_series).all():
            metrics_values = {metric: np.nan for metric in metrics_dict}
        else:
            metrics_values = {
                'KGE': kge_calc(refer_series, model_series)[0],
                'R': kge_calc(refer_series, model_series)[1],
                'ALPHA': kge_calc(refer_series, model_series)[2],
                'BETA': kge_calc(refer_series, model_series)[3],
                'TSS': tss_calc(refer_series, model_series),
                'NSE': nse_calc(refer_series, model_series),
                'RMSE': rmse_calc(refer_series, model_series),
                'NRMSE': nrmse_calc(refer_series, model_series),
                'MARE': mare_calc(refer_series, model_series),
                'PBIAS': pbias_calc(refer_series, model_series),
                'EVS': evs_calc(refer_series, model_series),
                'R2': r2_calc(refer_series, model_series),
                'MSE': mse_calc(refer_series, model_series),
                'MAE': mae_calc(refer_series, model_series),
                'MAPE': mape_calc(refer_series, model_series),
                'MBE': mbe_calc(refer_series, model_series),
                'PCC': pcc_calc(refer_series, model_series)[0],
                'PCC_PVAL': pcc_calc(refer_series, model_series)[1],
                'ACC': acc_calc(refer_series, model_series)[0],
                'ACC_PVAL': acc_calc(refer_series, model_series)[1],
                'TSTAT': ttest_calc(refer_series, model_series)[0],
                'TSTAT_PVAL': ttest_calc(refer_series, model_series)[1],
            }

        metrics_values_list.append(metrics_values)
    return lat_idx, metrics_values_list


def calculate_evalmetrics_spatial_mproc(exp_dict, save_path, data_path=None, varname='prec'):
    
    ref_data = exp_dict['REF'][0]
    print(ref_data)
    refer_data = xr.open_dataset(f'{data_path}/{ref_data}' if data_path is None else ref_data)
    refer_variable = refer_data[varname].load()

    # Create empty arrays for storage
    metrics_dict = {
        'KGE', 'R', 'ALPHA', 'BETA', 'TSS', 'NSE', 'RMSE', 'NRMSE', 'MARE', 'PBIAS', 
        'EVS', 'R2', 'MSE', 'MAE', 'MAPE', 'MBE', 
        'PCC', 'PCC_PVAL', 'ACC', 'ACC_PVAL',  'TSTAT', 'TSTAT_PVAL',
        }
    
    metrics_arrays = {metric: np.zeros_like(refer_variable.isel(time=0).values) for metric in metrics_dict}

    for exp_id, (exp_data, label) in exp_dict.items():
        print(f'Processing ... {exp_id}: {label}')

        if exp_id == 'REF':
            print('Skipping ...')
            continue

        model_data = xr.open_dataset(f'{data_path}/{exp_data}' if data_path is None else exp_data)
        model_variable = model_data[varname].load()

        # # Use multiprocessing to parallelize latitude processing with a progress bar
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     # Create a tqdm progress bar
        #     results = list(tqdm(pool.starmap(process_latitude, 
        #                         [(lat_idx, refer_variable, model_variable, metrics_dict) 
        #                             for lat_idx in range(refer_variable.shape[1])]),
        #                     total=refer_variable.shape[1],  # Set total to the number of latitudes
        #                     desc=f'Processing {exp_id}'))  # Optional: description of the task
        

        # Use multiprocessing to parallelize latitude processing
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(process_latitude, 
                                   [(lat_idx, refer_variable, model_variable, metrics_dict) 
                                    for lat_idx in range(refer_variable.shape[1])])

        # Collect results and assign them to arrays
        for lat_idx, metrics_values_list in results:
            for lon_idx, metrics_values in enumerate(metrics_values_list):
                for metric, value in metrics_values.items():
                    metrics_arrays[metric][lat_idx, lon_idx] = value

        # Save logic remains the same
        try:
            # Create xarray datasets for each metric
            datasets = {metric: xr.Dataset({metric: (['lat', 'lon'], values)}, 
                                        coords={'lat': model_variable['lat'],  
                                                'lon': model_variable['lon']})
                        for metric, values in metrics_arrays.items()}

            # Combine datasets into a single dataset
            combined_data = xr.Dataset({f'{exp_id}_{metric.upper()}': dataset[metric] for metric, dataset in datasets.items()},
                                    coords={'lat': model_variable['lat'],       
                                            'lon': model_variable['lon']
                                            })

            # Save the combined dataset to the same NetCDF file
            output_file_path = f'{save_path}/TEMP2MERGE_{exp_id}.nc'
            combined_data.to_netcdf(output_file_path, mode='w', format='NETCDF4')
            print(f'\nCreated {output_file_path}')
        
        except Exception as e:
            print(f"Failed to save NetCDF file: {e}")
            # Save as pickle file if NetCDF save fails
            pickle_output_file_path = f'{save_path}/TEMP2MERGE_{exp_id}.pkl'
            with open(pickle_output_file_path, 'wb') as f:
                pickle.dump(combined_data, f)
            print(f"Saved as pickle object: {pickle_output_file_path}")


##########################################################################################


#%% Run the code below


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

SAVE_PATH = "/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/PLTDATA/EVAL_METRICS_SPATIAL"

if __name__ == "__main__":
    create_directory(SAVE_PATH)
    calculate_evalmetrics_spatial(data_dict, data_path=DATA_PATH, save_path=SAVE_PATH, varname='prec')













