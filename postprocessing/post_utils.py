#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 22:23:51 2024

@author: midhunm
"""

import os
import datetime as dt
from netCDF4 import Dataset
from netCDF4 import date2num #, num2date
import numpy as np
import xarray as xr
import argparse
import tensorflow as tf

def clean_data(arr, clipmax = 1200):
    arr = np.where(arr<1, 0, arr)
    arr = np.clip(arr, None, clipmax)
    return arr

def negtozero(arr):
    return np.where(arr < 0, 0, arr)

def r_logtrans(arr):
    return 10**(arr) - 1

def load_inputs(inputs_channels: dict,
                bound=None, 
                chunk = 'after',
                ):
    """
    Data Loader for Phy-SRDRN
    """
    
    # Select Inputs Channels
    sel_inputs_channels = {}
    for channel, npy_path in inputs_channels.items():
        print(f'\tLoading Inputs Channel ... {channel}: {npy_path}')
        sel_inputs_channels[channel] = np.load(npy_path).squeeze()
        print(f'\tShape: {sel_inputs_channels[channel].shape}')
    
    # Stack the input arrays
    inputs = np.stack(list(sel_inputs_channels.values()), axis=3)
    
    if chunk=='after':
        inputs = inputs if bound is None else inputs[bound:]
        print(f"\n\tFinish Process Inputs Channels: {inputs.shape}")
            
        return inputs
    
    elif chunk=='before':
        inputs = inputs if bound is None else inputs[:bound]
        print(f"\n\tFinish Process Inputs Channels: {inputs.shape}")
            
        return inputs

def create_netcdf_from_array(array,
                             varname,
                             ref_ds, 
                             start_date = "1981-01-01",
                             end_date = "2020-12-31",
                             filename='newfile.nc',
                             ):
    
    def extract_date_components(date_string):
        year, month, day = map(int, date_string.split("-"))
        return year, month, day
    
    def find_lat_lon_vars(ds):

        # Initialize variables for latitude and longitude
        lat_var = None
        lon_var = None

        # Iterate through variable names to find latitude and longitude
        for var_name in ds.variables:
            var = ds[var_name]
            if hasattr(var, "standard_name"):
                if var.standard_name.lower() in ["latitude", "lat"]:
                    lat_var = var_name
                elif var.standard_name.lower() in ["longitude", "lon"]:
                    lon_var = var_name
            elif var_name.lower() in ["latitude", "lat"]:
                lat_var = var_name
            elif var_name.lower() in ["longitude", "lon"]:
                lon_var = var_name

        # Close the NetCDF file
        ds.close()
        
        print(f'\tLat.Var: {lat_var}, Lon.Var: {lon_var}')

        return lat_var, lon_var
    
    def date_range(start, end):
        delta = end - start  # as timedelta
        days = [start + dt.timedelta(days=i) for i in range(delta.days + 1)]
        return days
    
    if os.path.exists(filename):
        os.remove(filename)
    
    lat_var, lon_var = find_lat_lon_vars(ref_ds)
    lat_values, lon_values = ref_ds[lat_var].data, ref_ds[lon_var].data
    
    # -------------------
    # Creating dimensions
    # -------------------
    
    ncfile = Dataset(filename, mode='w', format='NETCDF4')
    print(ncfile)
    
    ncfile.createDimension('lat', len(lat_values))    # latitude axis
    ncfile.createDimension('lon', len(lon_values))    # longitude axis
    ncfile.createDimension('time', None)  # unlimited axis (can be appended to).
    for dim in ncfile.dimensions.items():
        print(dim)

    # Creating attributes
    ncfile.title = os.path.basename(filename)
    print(ncfile.title)

    ncfile.subtitle="Processed data"
    print(ncfile.subtitle)
    print(ncfile)
    
    # -------------------
    # Creating  variables
    # -------------------
    
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = f'days since {start_date}'
    time.long_name = 'time'

    var = ncfile.createVariable(varname, np.float64,('time','lat','lon')) # Define a 3D variable to hold the data
    var.units = 'unit'
    # var.standard_name = 'data'
    print(var)
    print("-- Some pre-defined attributes for variable temp:")
    print("rf.dimensions:", var.dimensions)
    print("rf.shape:", var.shape)
    print("rf.dtype:", var.dtype)
    print("rf.ndim:", var.ndim)
    
    # -------------
    # Writing data
    # -------------
    
    lat[:] = lat_values 
    lon[:] = lon_values 
    var[:,:,:] = array

    print(time)
    times_arr = time[:]
    print(type(times_arr),times_arr)
    
    s_year, s_mon, s_day = extract_date_components(start_date)
    e_year, e_mon, e_day = extract_date_components(end_date)

    dates = date_range(dt.datetime(s_year, s_mon, s_day), dt.datetime(e_year, e_mon, e_day))
    times = date2num(dates, time.units)
    time[:] = times
    print(time[:])
    print(time.units)

    print(ncfile) 
    # print(num2date(time[:]))
    ncfile.close() 
    print(f'Dataset is created: {filename}') # close the Dataset.
    
# def generate_data_test(prefix,
#                        model_path, 
#                        inputs_arr, 
#                        ref_data_path = 'reference.nc',
#                        save_dir = '.',
#                        save_nc = True,
#                        save_npy = True,
#                        ):
    
#     rd = xr.open_dataset(ref_data_path)
    
#     model = tf.keras.models.load_model(model_path, compile=False)
    
#     y_pred = r_logtrans(negtozero(model.predict(inputs_arr, batch_size=32)))
#     print('#'*10, f"Maximum value: {np.max(y_pred)}", '#'*10)
    
#     if save_nc:
#         create_netcdf_from_array(y_pred.squeeze(),
#                                  varname = 'prec',
#                                  ref_ds = rd,
#                                  start_date = dt.datetime(2015, 1, 1),
#                                  end_date = dt.datetime(2020, 12, 31),
#                                  filename = f'{save_dir}/{prefix}_012_out.nc',
#                                  )
#     if save_npy:
#         np.save(f'{save_dir}/{prefix}_testdata.npy', y_pred.squeeze())

def generate_data_test_from_numpy(prefix,
                                  array, 
                                  ref_data_path,
                                  start_date = "1980-01-01",
                                  end_date = "2020-12-31",
                                  save_dir = '.',
                                  save_nc = True,
                                  save_npy = False,
                                  varname = "prec",
                                  suffix = None,
                                  ):
    
    rd = xr.open_dataset(ref_data_path)
    
    if save_nc:
        create_netcdf_from_array(array.squeeze(),
                                 varname = varname,
                                 ref_ds = rd,
                                 start_date = start_date,
                                 end_date = end_date,
                                 filename = f"{save_dir}/{prefix}_{suffix}.nc" if suffix is not None else f"{save_dir}/{prefix}.nc",
                                 )
    if save_npy:
        np.save(f'{save_dir}/{prefix}_{suffix}.npy', array.squeeze())
