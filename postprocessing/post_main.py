#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 22:20:51 2024

@author: midhunm
"""
#%% Create dictionary of inputs combinations

import numpy as np
from post_utils import generate_data_test_from_numpy, r_logtrans, negtozero, clean_data


SAVE_PATH = '/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/GENDATA'

REF_DATA = '/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/GENDATA/raw/ref_imda_012_data.nc'

DATA_PATH = '/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/RAWRUNS/P07A.N123-S123.L12.I123.B1.R1'
# DATA_PATH = '/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/RAWRUNS/P07A.U12.L12.I123.B1.R1'

prefix_list = [
    # 'p07a_n01_l01_i01_b01_r01',
    # 'p07a_n01_l01_i02_b01_r01',
    # 'p07a_n01_l01_i03_b01_r01',
    # 'p07a_n01_l02_i01_b01_r01',
    # 'p07a_n01_l02_i02_b01_r01',
    # 'p07a_n01_l02_i03_b01_r01',
    # 'p07a_n02_l01_i01_b01_r01',
    # 'p07a_n02_l01_i02_b01_r01',
    # 'p07a_n02_l01_i03_b01_r01',
    # 'p07a_n02_l02_i01_b01_r01',
    # 'p07a_n02_l02_i02_b01_r01',
    # 'p07a_n02_l02_i03_b01_r01',
    # 'p07a_n03_l01_i01_b01_r01',
    # 'p07a_n03_l01_i02_b01_r01',
    # 'p07a_n03_l01_i03_b01_r01',
    # 'p07a_n03_l02_i01_b01_r01',
    # 'p07a_n03_l02_i02_b01_r01',
    # 'p07a_n03_l02_i03_b01_r01',
    # 'p07a_s01_l01_i01_b01_r01',
    # 'p07a_s01_l01_i02_b01_r01',
    # 'p07a_s01_l01_i03_b01_r01',
    # 'p07a_s01_l02_i01_b01_r01',
    # 'p07a_s01_l02_i02_b01_r01',
    # 'p07a_s01_l02_i03_b01_r01',
    # 'p07a_s02_l01_i01_b01_r01',
    # 'p07a_s02_l01_i02_b01_r01',
    # 'p07a_s02_l01_i03_b01_r01',
    # 'p07a_s02_l02_i01_b01_r01',
    # 'p07a_s02_l02_i02_b01_r01',
    # 'p07a_s02_l02_i03_b01_r01',
    # 'p07a_s03_l01_i01_b01_r01',
    # 'p07a_s03_l01_i02_b01_r01',
    # 'p07a_s03_l01_i03_b01_r01',
    # 'p07a_s03_l02_i01_b01_r01',
    # 'p07a_s03_l02_i02_b01_r01',
    # 'p07a_s03_l02_i03_b01_r01',

    # 'p07a_u01_l01_i01_b01_r01',
    # 'p07a_u01_l01_i02_b01_r01',
    # 'p07a_u01_l01_i03_b01_r01',
    # 'p07a_u01_l02_i01_b01_r01',
    # 'p07a_u01_l02_i02_b01_r01',
    # 'p07a_u01_l02_i03_b01_r01',
    # 'p07a_u02_l01_i01_b01_r01',
    # 'p07a_u02_l01_i02_b01_r01',
    # 'p07a_u02_l01_i03_b01_r01',
    # 'p07a_u02_l02_i01_b01_r01',
    # 'p07a_u02_l02_i02_b01_r01',
    # 'p07a_u02_l02_i03_b01_r01',
    ] 

for prefix in prefix_list:

    print(f'\n Processing ... {prefix}')
    
    y_pred = r_logtrans(negtozero(np.load(f"{DATA_PATH}/{prefix}_out_raw.npy"))).astype(np.float32)
    y_pred = clean_data(y_pred)
    
    generate_data_test_from_numpy(prefix = prefix,
                                  array = y_pred, 
                                  start_date = "2015-01-01",
                                  end_date = "2020-12-31",
                                  ref_data_path = REF_DATA,
                                  save_dir = SAVE_PATH,
                                  suffix = "012_out"
                                  )

    print(f"Max Pred: {y_pred.max()}")
    print('**'*60)
    

#%%
    
import numpy as np
import xarray as xr
from post_utils import generate_data_test_from_numpy, r_logtrans, negtozero, clean_data

REF_DATA = '/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/GENDATA/raw/ref_imda_012_data.nc'
SAVE_PATH = '/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/GENDATA'

DATA_PATH = '/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/GENDATA/raw'

prefix_list = [
    'quantile_mapping_disagregated_out',
    'ref_imda_012_data',
    ] 

for prefix in prefix_list:

    print(f'\n Processing ... {prefix}')

    y = xr.open_dataset(f"{DATA_PATH}/{prefix}.nc").prec.data.squeeze()
    y = clean_data(y)
    
    generate_data_test_from_numpy(prefix = prefix,
                                  array = y, 
                                  start_date = "2015-01-01",
                                  end_date = "2020-12-31",
                                  ref_data_path = REF_DATA,
                                  save_dir = SAVE_PATH,
                                  )

    print(f"Max Pred: {y.max()}")
    print('**'*60)
    


# %%
