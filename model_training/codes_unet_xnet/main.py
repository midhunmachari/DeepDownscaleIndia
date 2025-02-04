"""
author: Midhun Murukesh

Experiment description: DeepDown Intercomparison
"""

#%%

# ---------------------------------
#         LOOP STRUCTURE
# ---------------------------------
# Loop 1 : Learning Rate (r)
# Loop 2 : Batch Size (b)
# Loop 3 : Inputs Combinations (i)
# Loop 4 : Loss Function (l)
# Loop 5 : Model Architecture (nsu)
# ---------------------------------

import argparse
import tensorflow as tf

from train import ModelTraining
from utils import load_inputs_target_pairs, train_val_split
from models import UNET, NEST_UNET
from losses import weighted_mae, BernoulliGammaLoss

tf.keras.backend.clear_session()
    
# Create an argument parser
parser = argparse.ArgumentParser()

# Add the epoch argument to the parser
parser.add_argument('--epochs', type=int, default=10   , help='Number of epochs to train')
parser.add_argument('--path'  , type=str, default='./' , help='Path to present working directory')
parser.add_argument('--dpath' , type=str, default='./' , help='path to data file in NPZ')
parser.add_argument('--prefix', type=str, default='rdx', help='Prefix of current experiment')

# Parse the command-line arguments
args = parser.parse_args()
DPATH = args.dpath

#%% The training block
print('#'*100)

# -------------------------  
# Loop1: Learrning rate: r
# -------------------------  

learning_rate_dict = {
    'r01' : 1e-4,
    # 'r02' : 2e-4,
    # 'r04' : 5e-4,
    }

for lr_id, lr in learning_rate_dict.items():

##### ---------------------
##### Loop2: Batch size: r
##### ---------------------
    
    batch_size_dict = {
         'b01' : 16,
         # 'b02' : 32,
        }
    
    for bs_id, bs in batch_size_dict.items():
        
######### -----------------------------
######### Loop3: Inputs combination: b
######### -----------------------------
        
        i01 = {
            'era5_prec_100_logn': f'{DPATH}/IND32_CH1_100_ERA5_PREC_DAY_1980_2020_LOGN.npy',
               }

        i02 = {
            
            'era5_h850_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_H850_DAY_1980_2020_STDN.npy',
            'era5_h700_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_H700_DAY_1980_2020_STDN.npy',
            'era5_h500_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_H500_DAY_1980_2020_STDN.npy',
            'era5_h250_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_H250_DAY_1980_2020_STDN.npy',
            
            'era5_q850_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_Q850_DAY_1980_2020_STDN.npy',
            'era5_q700_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_Q700_DAY_1980_2020_STDN.npy',
            'era5_q500_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_Q500_DAY_1980_2020_STDN.npy',
            'era5_q250_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_Q250_DAY_1980_2020_STDN.npy',
            
            'era5_t850_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_T850_DAY_1980_2020_STDN.npy',
            'era5_t700_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_T700_DAY_1980_2020_STDN.npy',
            'era5_t500_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_T500_DAY_1980_2020_STDN.npy',
            'era5_t250_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_T250_DAY_1980_2020_STDN.npy',
            
            'era5_u850_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_U850_DAY_1980_2020_STDN.npy',
            'era5_u700_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_U700_DAY_1980_2020_STDN.npy',
            'era5_u500_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_U500_DAY_1980_2020_STDN.npy',
            'era5_u250_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_U250_DAY_1980_2020_STDN.npy',
            
            'era5_v850_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_V850_DAY_1980_2020_STDN.npy',
            'era5_v700_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_V700_DAY_1980_2020_STDN.npy',
            'era5_v500_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_V500_DAY_1980_2020_STDN.npy',
            'era5_v250_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_V250_DAY_1980_2020_STDN.npy',

        }
        
        i03 = {

            'era5_prec_100_logn': f'{DPATH}/IND32_CH1_100_ERA5_PREC_DAY_1980_2020_LOGN.npy',
            
            'era5_h850_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_H850_DAY_1980_2020_STDN.npy',
            'era5_h700_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_H700_DAY_1980_2020_STDN.npy',
            'era5_h500_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_H500_DAY_1980_2020_STDN.npy',
            'era5_h250_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_H250_DAY_1980_2020_STDN.npy',
            
            'era5_q850_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_Q850_DAY_1980_2020_STDN.npy',
            'era5_q700_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_Q700_DAY_1980_2020_STDN.npy',
            'era5_q500_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_Q500_DAY_1980_2020_STDN.npy',
            'era5_q250_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_Q250_DAY_1980_2020_STDN.npy',
            
            'era5_t850_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_T850_DAY_1980_2020_STDN.npy',
            'era5_t700_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_T700_DAY_1980_2020_STDN.npy',
            'era5_t500_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_T500_DAY_1980_2020_STDN.npy',
            'era5_t250_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_T250_DAY_1980_2020_STDN.npy',
            
            'era5_u850_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_U850_DAY_1980_2020_STDN.npy',
            'era5_u700_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_U700_DAY_1980_2020_STDN.npy',
            'era5_u500_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_U500_DAY_1980_2020_STDN.npy',
            'era5_u250_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_U250_DAY_1980_2020_STDN.npy',
            
            'era5_v850_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_V850_DAY_1980_2020_STDN.npy',
            'era5_v700_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_V700_DAY_1980_2020_STDN.npy',
            'era5_v500_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_V500_DAY_1980_2020_STDN.npy',
            'era5_v250_100_stdn': f'{DPATH}/IND32_CHX_100_ERA5_V250_DAY_1980_2020_STDN.npy',

        }
        
        
        inputs_dict = {
            'i01': i01,
            'i02': i02,
            'i03': i03,
            }

        static_channels = {
             'imda_dclm_012_logn' : f'{DPATH}/IND32_CH2_012_IMDA_DCLM_DAY_1980_2020_LOGN.npy',
             'usgs_elev_012_logn' : f'{DPATH}/IND32_CH3_012_USGS_ELEV_DAY_1980_2020_LOGN.npy',
             }

        target_channels = {
             'imda_prec_012_log'  : f'{DPATH}/IND32_CH0_012_IMDA_PREC_DAY_1980_2020_LOG.npy',
             }

        for inputs_id, inputs_channels in inputs_dict.items():
            
            # Load the dataset for model training and validation
            X, S, y = load_inputs_target_pairs(inputs_channels,
                                               target_channels,
                                               static_channels,
                                               )
            
            # Assuming X and S have the same number of samples
            
            if inputs_id=='i02':
                X_train, X_val, y_train, y_val = train_val_split(X, y)
                print(f"X_train shape: {X_train.shape}, X_train max.: {X_train.max()}")
                print(f"X_val shape: {X_val.shape}, X_val max.: {X_val.max()}")
                print(f"y_train shape: {y_train.shape}, y_train max.: {y_train.max()}")
                print(f"y_val shape: {y_val.shape}, y_val max.: {y_val.max()}")
            
            else:
                X_train, X_val, S_train, S_val, y_train, y_val = train_val_split(X, y, S)
            
                print(f"X_train shape: {X_train.shape}, X_train max.: {X_train.max()}")
                print(f"X_val shape: {X_val.shape}, X_val max.: {X_val.max()}")
                print(f"S_train shape: {S_train.shape}, S_train max.: {S_train.max()}")
                print(f"S_val shape: {S_val.shape},S_val max.: {S_val.max()}")
                print(f"y_train shape: {y_train.shape}, y_train max.: {y_train.max()}")
                print(f"y_val shape: {y_val.shape}, y_val max.: {y_val.max()}")
            
############# -------------------------------
############# Loop4: Loss fn. combination: l
############# -------------------------------
        
            losses_dict = {
                
                'l01': weighted_mae,
                'l02': BernoulliGammaLoss(),
                
                }
            
            for loss_id, loss in losses_dict.items():
                
##################### -------------------------------------
##################### Loop5: Model Arch.. combination: nsu
##################### -------------------------------------
                    
                    u01 = UNET(lr_input_shape = X_train.shape[1:],
                               hr_input_shape = None if inputs_id=='i02' else S_train.shape[1:],
                               ups_factors = (2,2,2),
                               layer_N=[64, 96, 128, 160],
                               input_stack_num=2,
                               pool=True,
                               activation='prelu',
                               last_kernel_size = 1,
                               isgammaloss = True if loss_id=='l02' else False,
                               )

                    u02 = NEST_UNET(lr_input_shape = X_train.shape[1:],
                               hr_input_shape = None if inputs_id=='i02' else S_train.shape[1:],
                               ups_factors = (2,2,2),
                               layer_N=[64, 96, 128, 160],
                               input_stack_num=2,
                               pool=True,
                               activation='prelu',
                               last_kernel_size = 1,
                               isgammaloss = True if loss_id=='l02' else False,
                                )

                    models_dict = {
                        
                        'u01': u01,
                        'u02': u02,
                        
                        }
                
                    for model_id, model_arch in models_dict.items():
                    
                        prefix = f"{args.prefix}_{model_id}_{loss_id}_{inputs_id}_{bs_id}_{lr_id}"
                        print(f'\nInitiate experiment: {prefix}')
                            
                        print('Configuring model architecture')
                        model = model_arch
                        
                        print('#### MODEL SUMMARY ####')
                        print(model.summary())
                        
                        mt = ModelTraining(prefix=prefix, pwd=args.path)
                            
                        mt.train(model,
                                 train_data = (X_train, y_train) if inputs_id=='i02' else (X_train, S_train, y_train),
                                 valid_data = (X_val, y_val) if inputs_id=='i02' else (X_val, S_val, y_val),
                                 epochs = args.epochs, 
                                 loss = loss,         
                                 batch_size = bs,      
                                 learning_rate = lr,
                                )
                            
                        if inputs_id=='i02':
                            X_test, y_test = train_val_split(X, y, test_only=True)
                            mt.gen_predictions([X_test], loss_fn="gamma" if loss_id=='l02' else None)
                        else:  
                            X_test, S_test, y_test = train_val_split(X, y, S, test_only=True)
                            mt.gen_predictions([X_test, S_test], loss_fn="gamma" if loss_id=='l02' else None)
                                
