#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:22:40 2023

@author: midhunm
"""
import numpy as np
import tensorflow as tf
# from tensorflow import keras

#%% n01: SIMPLE_DENSE

def LINEAR_DENSE(input_shape,
                 ups_factors, 
                 n_neurons = [256],
                 output_shape = (320,320,1),
                 dropout=0.5,
                 isgammaloss = False,
                 ):
    """
    Create a simple dense neural network model.

    Args:
        dense_layers (list): List of dense layer sizes.
        dense_activation (str): Activation function for dense layers.
        input_shape (tuple): Input shape of the model.
        dropout (float): Dropout rate.

    Returns:
        tf.keras.Model: The constructed dense neural network model.

    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = input_dense(inputs, dropout=dropout)
    for neuron in n_neurons:
        x = tf.keras.layers.Dense(neuron)(x)
        x = tf.keras.layers.PReLU()(x)
    if isgammaloss:
        out1 = tf.keras.layers.Dense(np.prod(output_shape), activation='selu')(x)
        out1 = tf.keras.layers.Reshape((np.prod(ups_factors)*input_shape[0],np.prod(ups_factors)*input_shape[1],1))(out1)
        out2 = tf.keras.layers.Dense(np.prod(output_shape), activation='selu')(x)
        out2 = tf.keras.layers.Reshape((np.prod(ups_factors)*input_shape[0],np.prod(ups_factors)*input_shape[1],1))(out2)
        out3 = tf.keras.layers.Dense(np.prod(output_shape), activation='sigmoid')(x)
        out3 = tf.keras.layers.Reshape((np.prod(ups_factors)*input_shape[0],np.prod(ups_factors)*input_shape[1],1))(out3)
        out = tf.keras.layers.Concatenate()([out1, out2, out3])
        return tf.keras.models.Model(inputs, out, name='LINEAR-DENSE-x3')
    else:  
        x = tf.keras.layers.Dense(np.prod(output_shape))(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.Reshape((np.prod(ups_factors)*input_shape[0],np.prod(ups_factors)*input_shape[1],1))(x)
        return tf.keras.models.Model(inputs, x, name='LINEAR-DENSE')


#%% n02: CONV_DENSE

def CONV_DENSE(input_shape,
                ups_factors,
                layer_filters=[16, 64, 128], 
                bn=True, 
                padding='same', 
                kernel_size=3,
                pooling=True, 
                dense_layers=[256], 
                dense_activation=tf.keras.layers.PReLU(),
                dropout=0.5, 
                activation=tf.keras.layers.PReLU(),
                isgammaloss = False,
                ):
    """
    Create a complex convolutional neural network model.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = contruct_base_conv(inputs, layer_filters=layer_filters, bn=bn, padding=padding,
                         kernel_size=kernel_size, pooling=pooling, activation=activation, dropout=dropout)
    for neuron in dense_layers:
        if activation=='prelu':
            x = tf.keras.layers.Dense(neuron)(x)
            x = tf.keras.layers.PReLU()(x)
        else:
            x = tf.keras.layers.Dense(neuron, activation=activation)(x)
        # x = tf.keras.layers.Dropout(dropout)(x)
    if isgammaloss:
        out1 = tf.keras.layers.Dense(np.prod(ups_factors)*input_shape[0]*np.prod(ups_factors)*input_shape[1], 
                                     activation='selu', kernel_initializer='zeros')(x)
        out1 = tf.keras.layers.Reshape((np.prod(ups_factors)*input_shape[0],np.prod(ups_factors)*input_shape[1],1))(out1)
        out2 = tf.keras.layers.Dense(np.prod(ups_factors)*input_shape[0]*np.prod(ups_factors)*input_shape[1], 
                                     activation='selu', kernel_initializer='zeros')(x)
        out2 = tf.keras.layers.Reshape((np.prod(ups_factors)*input_shape[0],np.prod(ups_factors)*input_shape[1],1))(out2)
        out3 = tf.keras.layers.Dense(np.prod(ups_factors)*input_shape[0]*np.prod(ups_factors)*input_shape[1], 
                                     activation='sigmoid', kernel_initializer='zeros')(x)
        out3 = tf.keras.layers.Reshape((np.prod(ups_factors)*input_shape[0],np.prod(ups_factors)*input_shape[1],1))(out3)
        out = tf.keras.layers.Concatenate()([out1, out2, out3])
        return tf.keras.models.Model(inputs, out, name='CONV-DENSE-x3')
    else:
        x = tf.keras.layers.Dense(np.prod(ups_factors)*input_shape[0]*np.prod(ups_factors)*input_shape[1], 
                                  activation='linear', kernel_initializer='zeros')(x)
        x = tf.keras.layers.Reshape((np.prod(ups_factors)*input_shape[0],np.prod(ups_factors)*input_shape[1],1))(x)
        return tf.keras.models.Model(inputs, x, name='CONV-DENSE')


#%% n03: FSRCNN

def FSRCNN(input_shape,
           ups_factors,
           k_size = 3, 
           n = 128,
           d = 64, 
           s = 32,
           m = 4, 
           isgammaloss = False,
           ):
    """
    FSRCNN model implementation from https://arxiv.org/abs/1608.00367
    
    Sigmoid Activation in the output layer is not in the original paper.
    But it is needed to align model prediction with ground truth HR images
    so their values are in the same range [0,1].
    """
    
    inputs = tf.keras.layers.Input(shape = input_shape)
    # feature extraction
    model = tf.keras.layers.Conv2D(kernel_size=5, filters=d, padding="same", kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    model = tf.keras.layers.PReLU(alpha_initializer="zeros", shared_axes=[1, 2])(model)
    # Shrinking
    model = tf.keras.layers.Conv2D(kernel_size=1, filters=s, padding="same", kernel_initializer=tf.keras.initializers.HeNormal())(model)
    model = tf.keras.layers.PReLU(alpha_initializer="zeros", shared_axes=[1, 2])(model)
    # Mapping
    for _ in range(m):
        model = tf.keras.layers.Conv2D(kernel_size=3, filters=s, padding="same", kernel_initializer=tf.keras.initializers.HeNormal())(model)
        model = tf.keras.layers.PReLU(alpha_initializer="zeros", shared_axes=[1, 2])(model)
    # Expanding
    model = tf.keras.layers.Conv2D(kernel_size=1, filters=d, padding="same")(model)
    model = tf.keras.layers.PReLU(alpha_initializer="zeros", shared_axes=[1, 2])(model)
    # Deconvolution
    for _, ups_size in enumerate(ups_factors):
        model =tf.keras.layers.Conv2DTranspose(kernel_size=3, filters=n, strides=ups_size, padding="same", 
                                             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.001))(model)
        model = tf.keras.layers.PReLU(alpha_initializer="zeros", shared_axes=[1, 2])(model)
    if isgammaloss:
        out1 = tf.keras.layers.Conv2D(kernel_size=k_size, filters=1, padding="same", 
                                   activation='selu', kernel_initializer=tf.keras.initializers.HeNormal())(model)
        out2 = tf.keras.layers.Conv2D(kernel_size=k_size, filters=1, padding="same", 
                                   activation='selu', kernel_initializer=tf.keras.initializers.HeNormal())(model)
        out3 = tf.keras.layers.Conv2D(kernel_size=k_size, filters=1, padding="same", 
                                   activation='sigmoid', kernel_initializer=tf.keras.initializers.HeNormal())(model)
        out = tf.keras.layers.Concatenate()([out1, out2, out3])
        return tf.keras.models.Model(inputs, out, name='FSRCNN-x3')
    else:
        model = tf.keras.layers.Conv2D(kernel_size=k_size, filters=1, padding="same", 
                                       kernel_initializer=tf.keras.initializers.HeNormal())(model)
        return tf.keras.models.Model(inputs, model, name='FSRCNN')

#%% s01: SRDRN

def SRDRN(input_shape,
          ups_factors,
          n_filters = 64,
          n_res_blocks = 16, 
          n_ups_filters = 256,
          last_kernel_size = 3,
          activation = 'prelu',
          regularizer = tf.keras.regularizers.l2(0.01),
          initializer = tf.keras.initializers.RandomNormal(stddev=0.02), 
          interpolation='nearest',
          isgammaloss = False,
          ):
    
    """
    Constructs a Super-resolution Residual Deep Neural Network (SRDRN) generator model (Wang et al. 2020).
    """
    
    inputs = tf.keras.layers.Input(shape = input_shape)
    model = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = 3, strides = 1, padding = "same", 
                                kernel_regularizer=regularizer, kernel_initializer=initializer)(inputs)
    model = SRDRN_activation(model, activation=activation)
    gen_model = model
    # Using 16 Residual Blocks
    for _ in range(n_res_blocks):
        model = SRDRN_residual_block(model, kernal_size=3, n_filters=n_filters, strides=1, 
                                     regularizer=regularizer, initializer=initializer, activation=activation)
    model = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = 3, strides = 1, padding = "same", 
                                kernel_initializer=initializer)(model)
    model = tf.keras.layers.BatchNormalization(momentum = 0.5)(model)
    model = tf.keras.layers.add([gen_model, model])
    for _, ups_size in enumerate(ups_factors):
        model = SRDRN_upsampling_block(model, ups_size=ups_size, n_filters=n_ups_filters, 
                                       activation=activation, regularizer=regularizer, initializer=initializer, 
                                       interpolation=interpolation)  
    if isgammaloss:
        out1 = tf.keras.layers.Conv2D(filters = 1, kernel_size = last_kernel_size, strides = 1, activation='selu', 
                                  padding = "same", kernel_regularizer=regularizer, 
                                  kernel_initializer=initializer)(model)
        out2 = tf.keras.layers.Conv2D(filters = 1, kernel_size = last_kernel_size, strides = 1, activation='selu', 
                                  padding = "same", kernel_regularizer=regularizer, 
                                  kernel_initializer=initializer)(model)
        out3 = tf.keras.layers.Conv2D(filters = 1, kernel_size = last_kernel_size, strides = 1, activation='sigmoid', 
                                  padding = "same", kernel_regularizer=regularizer, 
                                  kernel_initializer=initializer)(model)
        out = tf.keras.layers.Concatenate()([out1, out2, out3])
        return tf.keras.models.Model(inputs, out, name='SRDRN-x3')
    else:
        model = tf.keras.layers.Conv2D(filters = 1, kernel_size = last_kernel_size, strides = 1, activation='linear', 
                                  padding = "same", kernel_regularizer=regularizer, 
                                  kernel_initializer=initializer)(model)
        return tf.keras.models.Model(inputs, model, name='SRDRN')

#%% s02: SRDRN_TC

def SRDRN_TC(input_shape,
             ups_factors,
             n_filters = 64,
             n_res_blocks = 16, 
             n_ups_filters = 256,
             last_kernel_size = 3, 
             activation = 'prelu',
             regularizer = tf.keras.regularizers.l2(0.01),
             initializer = tf.keras.initializers.RandomNormal(stddev=0.02),
             isgammaloss = False,
             ):
    
    """
    Constructs a Super-resolution Residual Deep Neural Network (SRDRN) generator model (Wang et al. 2020).
    """
    
    inputs = tf.keras.layers.Input(shape = input_shape)
    model = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = 3, strides = 1, padding = "same", 
                                kernel_regularizer=regularizer, kernel_initializer=initializer)(inputs)
    model = SRDRN_activation(model, activation=activation)
    gen_model = model
    # Using 16 Residual Blocks
    for _ in range(n_res_blocks):
        model = SRDRN_residual_block(model, kernal_size=3, n_filters=n_filters, strides=1, 
                                     regularizer=regularizer, initializer=initializer, activation=activation)
    model = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = 3, strides = 1, padding = "same", 
                                kernel_initializer=initializer)(model)
    model = tf.keras.layers.BatchNormalization(momentum = 0.5)(model)
    model = tf.keras.layers.add([gen_model, model])
    for _, ups_size in enumerate(ups_factors):
        model = SRDRN_convtranspose_block(model, ups_size=ups_size, n_filters=n_ups_filters, 
                                          activation=activation, regularizer=regularizer, 
                                          initializer=initializer)  
    if isgammaloss:
        out1 = tf.keras.layers.Conv2D(filters = 1, kernel_size = last_kernel_size, strides = 1, activation='selu', 
                                  padding = "same", kernel_regularizer=regularizer, 
                                  kernel_initializer=initializer)(model)
        out2 = tf.keras.layers.Conv2D(filters = 1, kernel_size = last_kernel_size, strides = 1, activation='selu', 
                                  padding = "same", kernel_regularizer=regularizer, 
                                  kernel_initializer=initializer)(model)
        out3 = tf.keras.layers.Conv2D(filters = 1, kernel_size = last_kernel_size, strides = 1, activation='sigmoid', 
                                  padding = "same", kernel_regularizer=regularizer, 
                                  kernel_initializer=initializer)(model)
        out = tf.keras.layers.Concatenate()([out1, out2, out3])
        return tf.keras.models.Model(inputs, out, name='SRDRN-TC-x3')
    else:
        model = tf.keras.layers.Conv2D(filters = 1, kernel_size = last_kernel_size, strides = 1, activation='linear', 
                                  padding = "same", kernel_regularizer=regularizer, 
                                  kernel_initializer=initializer)(model)
        return tf.keras.models.Model(inputs, model, name='SRDRN-TC')


#%% s03: SRDRN_DENSE

def SRDRN_DENSE(input_shape,
                ups_factors,
                n_filters = 64,
                n_res_blocks = 16, 
                n_ups_filters = 256,
                n_dense=256,
                dropout = 0.5,
                activation = 'prelu',
                regularizer = tf.keras.regularizers.l2(0.01),
                initializer = tf.keras.initializers.RandomNormal(stddev=0.02), 
                isgammaloss = False,
                ):
    
    """
    Constructs a Super-resolution Residual Deep Neural Network (SRDRN) generator model (Wang et al. 2020).
    """
    
    inputs = tf.keras.layers.Input(shape = input_shape)
    model = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = 3, strides = 1, padding = "same", 
                                kernel_regularizer=regularizer, kernel_initializer=initializer)(inputs)
    model = SRDRN_activation(model, activation=activation)
 
    gen_model = model
    
    # Using 16 Residual Blocks
    for _ in range(n_res_blocks):
        model = SRDRN_residual_block(model, kernal_size=3, n_filters=n_filters, strides=1, 
                                     regularizer=regularizer, initializer=initializer, activation=activation)
    
    model = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = 3, strides = 1, padding = "same", 
                                kernel_initializer=initializer)(model)
    model = tf.keras.layers.BatchNormalization(momentum = 0.5)(model)
    model = tf.keras.layers.add([gen_model, model])
    
    model = tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = "same",
                            kernel_regularizer=regularizer, kernel_initializer=initializer)(model)
    model = SRDRN_activation(model, activation=activation)
    
    # Initiate Dense layers
    model = tf.keras.layers.Flatten()(model)
    model = tf.keras.layers.Dense(n_dense)(model)
    model = tf.keras.layers.PReLU()(model)
    model = tf.keras.layers.Dropout(dropout)(model)
    
    if isgammaloss:
        out1 = tf.keras.layers.Dense(np.prod(ups_factors)*input_shape[0]*np.prod(ups_factors)*input_shape[1], 
                                     activation='selu', kernel_initializer='zeros')(model)
        out1 = tf.keras.layers.Reshape((np.prod(ups_factors)*input_shape[0],np.prod(ups_factors)*input_shape[1],1))(out1)
        out2 = tf.keras.layers.Dense(np.prod(ups_factors)*input_shape[0]*np.prod(ups_factors)*input_shape[1], 
                                     activation='selu', kernel_initializer='zeros')(model)
        out2 = tf.keras.layers.Reshape((np.prod(ups_factors)*input_shape[0],np.prod(ups_factors)*input_shape[1],1))(out2)
        out3 = tf.keras.layers.Dense(np.prod(ups_factors)*input_shape[0]*np.prod(ups_factors)*input_shape[1], 
                                     activation='sigmoid', kernel_initializer='zeros')(model)
        out3 = tf.keras.layers.Reshape((np.prod(ups_factors)*input_shape[0],np.prod(ups_factors)*input_shape[1],1))(out3)
        out = tf.keras.layers.Concatenate()([out1, out2, out3])
        return tf.keras.models.Model(inputs, out, name='SRDRN-DENSE-x3')
    else:
        model = tf.keras.layers.Dense(np.prod(ups_factors)*input_shape[0]*np.prod(ups_factors)*input_shape[1], kernel_initializer='zeros')(model)
        model = tf.keras.layers.Activation("linear")(model)
        out = tf.keras.layers.Reshape((np.prod(ups_factors)*input_shape[0],np.prod(ups_factors)*input_shape[1],1))(model)
        return tf.keras.models.Model(inputs, out, name='SRDRN-DENSE')

#%% u01: UNET

def UNET(lr_input_shape, 
         hr_input_shape=None, 
         ups_factors=(2,2,2), 
         layer_N=[64, 96, 128, 160],
         input_stack_num=2, 
         pool=True, 
         activation='prelu',
         last_kernel_size = 1,
         isgammaloss = False,
         ):
    '''
    UNet with three down- and upsampling levels.
    '''
    
    IN_LR = tf.keras.layers.Input(lr_input_shape, name='unet_in_lr')
    
    x01 = IN_LR
    for _, ups_size in enumerate(ups_factors):
        x01 = tf.keras.layers.UpSampling2D(size=ups_size, interpolation='bilinear')(x01)
    if hr_input_shape is not None:
        # Concatenate the upsampled low resolution input and the high resolution input
        IN_HR = tf.keras.layers.Input(hr_input_shape, name='unet_in_hr')
        x = tf.keras.layers.concatenate([x01, IN_HR])
    else:
        x = x01
    X_en1 = CONV_stack(x, layer_N[0], kernel_size=3, stack_num=input_stack_num, activation=activation, name='unet_left0') 
    # downsampling levels
    X_en2 = UNET_left(X_en1, layer_N[1], pool=pool, activation=activation, name='unet_left1')
    X_en3 = UNET_left(X_en2, layer_N[2], pool=pool, activation=activation, name='unet_left2')
    X4 = UNET_left(X_en3, layer_N[3], pool=pool, activation=activation, name='unet_bottom')
    # upsampling levels
    X_de3 = UNET_right(X4, X_en3, layer_N[2], activation=activation, name='unet_right2')
    X_de2 = UNET_right(X_de3, X_en2, layer_N[1], activation=activation, name='unet_right1')
    X_de1 = UNET_right(X_de2, X_en1, layer_N[0], activation=activation, name='unet_right0')
    # output
    x = CONV_stack(X_de1, 16, kernel_size=3, stack_num=1, activation=activation, name='unet_out')
    if isgammaloss:
        out1 = tf.keras.layers.Conv2D(kernel_size=last_kernel_size, filters=1, 
                                      activation='selu', padding='same', name='unet_out1')(x)
        out2 = tf.keras.layers.Conv2D(kernel_size=last_kernel_size, filters=1, 
                                      activation='selu', padding='same', name='unet_out2')(x)
        out3 = tf.keras.layers.Conv2D(kernel_size=last_kernel_size, filters=1, 
                                      activation='sigmoid', padding='same', name='unet_out3')(x)
        OUT = tf.keras.layers.Concatenate()([out1, out2, out3])
        return tf.keras.models.Model(inputs=[IN_LR, IN_HR] if hr_input_shape is not None else [IN_LR], outputs=[OUT], name='UNET-x3')
    else:
        OUT = tf.keras.layers.Conv2D(kernel_size=last_kernel_size, filters=1, 
                                     activation="linear", padding='same', name='unet_exit')(x)
        return tf.keras.models.Model(inputs=[IN_LR, IN_HR] if hr_input_shape is not None else [IN_LR], outputs=[OUT], name='UNET')

#%% u02: NEST_UNET

def NEST_UNET(lr_input_shape, 
              hr_input_shape=None, 
              ups_factors=(2,2,2), 
              layer_N=[64, 96, 128, 160],
              input_stack_num=2,
              pool=True,
              activation='prelu',
              last_kernel_size=1,
              isgammaloss=False,
              ):
    '''
    Nest-UNet (or UNet++) with three down- and upsampling levels.
    '''
    # input layer
    IN_LR = tf.keras.layers.Input(lr_input_shape, name='unet_in_lr')
    
    x01 = IN_LR
    for _, ups_size in enumerate(ups_factors):
        x01 = tf.keras.layers.UpSampling2D(size=ups_size, interpolation='bilinear')(x01)
    if hr_input_shape is not None:
        # Concatenate the upsampled low resolution input and the high resolution input
        IN_HR = tf.keras.layers.Input(hr_input_shape, name='unet_in_hr')
        x = tf.keras.layers.concatenate([x01, IN_HR])    
    else:
        x = x01
    X11_conv = CONV_stack(x, layer_N[0], kernel_size=3, stack_num=input_stack_num, activation=activation, name='unet_left0')
    # downsampling levels (same as in the UNET)
    X21_conv = UNET_left(X11_conv, layer_N[1], pool=pool, activation=activation, name='unet_left1')
    X31_conv = UNET_left(X21_conv, layer_N[2], pool=pool, activation=activation, name='unet_left2')
    X41_conv = UNET_left(X31_conv, layer_N[3], pool=pool, activation=activation, name='unet_bottom')
    # up-sampling part 1
    X12_conv = XNET_right(X21_conv, [X11_conv], layer_N[0], activation=activation, name='xnet_12')
    X22_conv = XNET_right(X31_conv, [X21_conv], layer_N[1], activation=activation, name='xnet_22')
    X32_conv = XNET_right(X41_conv, [X31_conv], layer_N[2], activation=activation, name='xnet_32')
    # up-sampling part 2
    X13_conv = XNET_right(X22_conv, [X11_conv, X12_conv], layer_N[0], activation=activation, name='xnet_right13')
    X23_conv = XNET_right(X32_conv, [X21_conv, X22_conv], layer_N[1], activation=activation, name='xnet_right23')
    # up-sampling part 3
    X14_conv = XNET_right(X23_conv, [X11_conv, X12_conv, X13_conv], layer_N[0], activation=activation, name='xnet_right14')
    # output
    x =  CONV_stack(X14_conv, 16, kernel_size=3, stack_num=1, activation=activation, name='xnet_out')
    if isgammaloss:
        out1 = tf.keras.layers.Conv2D(kernel_size=last_kernel_size, filters=1, 
                                      activation='selu', padding='same', name='xnet_out1')(x)
        out2 = tf.keras.layers.Conv2D(kernel_size=last_kernel_size, filters=1, 
                                      activation='selu', padding='same', name='xnet_out2')(x)
        out3 = tf.keras.layers.Conv2D(kernel_size=last_kernel_size, filters=1, 
                                      activation='sigmoid', padding='same', name='xnet_out3')(x)
        OUT = tf.keras.layers.Concatenate()([out1, out2, out3])
        return tf.keras.models.Model(inputs=[IN_LR, IN_HR] if hr_input_shape is not None else [IN_LR], outputs=[OUT], name='NEST-UNET-x3')
    else:
        OUT = tf.keras.layers.Conv2D(kernel_size=last_kernel_size, filters=1, 
                                     activation="linear", padding='same', name='unet_exit')(x)
        return tf.keras.models.Model(inputs=[IN_LR, IN_HR] if hr_input_shape is not None else [IN_LR], outputs=[OUT], name='NEST-UNET')
    
#%% u03: UNET_DENSE

def UNET_DENSE(lr_input_shape, hr_input_shape, ups_factors, 
               layer_N=[64, 96, 128, 160],
               input_stack_num=2, 
               n_dense = 256,
               drop_rate = 0.5, 
               pool=True, 
               activation='prelu',
               ):
    '''
    UNet with three down- and upsampling levels.
    '''
    
    IN_LR = tf.keras.layers.Input(lr_input_shape, name='unet_in_lr')
    IN_HR = tf.keras.layers.Input(hr_input_shape, name='unet_in_hr')
    
    x01 = IN_LR
    for _, ups_size in enumerate(ups_factors):
        x01 = tf.keras.layers.UpSampling2D(size=ups_size, interpolation='bilinear')(x01)
    
    # Concatenate the upsampled low resolution input and the high resolution input
    x = tf.keras.layers.concatenate([x01, IN_HR])

    
    X_en1 = CONV_stack(x, layer_N[0], kernel_size=3, stack_num=input_stack_num, activation=activation, name='unet_left0')
    
    # downsampling levels
    X_en2 = UNET_left(X_en1, layer_N[1], pool=pool, activation=activation, name='unet_left1')
    X_en3 = UNET_left(X_en2, layer_N[2], pool=pool, activation=activation, name='unet_left2')
    
    X4 = UNET_left(X_en3, layer_N[3], pool=pool, activation=activation, name='unet_bottom')

    # upsampling levels
    X_de3 = UNET_right(X4, X_en3, layer_N[2], activation=activation, name='unet_right2')
    X_de2 = UNET_right(X_de3, X_en2, layer_N[1], activation=activation, name='unet_right1')
    X_de1 = UNET_right(X_de2, X_en1, layer_N[0], activation=activation, name='unet_right0')
    
    
    # output
    model = CONV_stack(X_de1, 16, kernel_size=3, stack_num=1, activation=activation, name='unet_out')
    # Initiate Dense layers
    model = tf.keras.layers.Conv2D(1,1)(model)
    model = tf.keras.layers.PReLU()(model)
    model = tf.keras.layers.Flatten()(model)
    model = tf.keras.layers.Dense(n_dense)(model)
    model = tf.keras.layers.PReLU()(model)
    model = tf.keras.layers.Dropout(drop_rate)(model)
    model = tf.keras.layers.Dense(np.prod(ups_factors)*lr_input_shape[0]*np.prod(ups_factors)*lr_input_shape[1])(model)
    model = tf.keras.layers.Activation("linear")(model)
    OUT =tf. keras.layers.Reshape((np.prod(ups_factors)*lr_input_shape[0], np.prod(ups_factors)*lr_input_shape[1],1))(model)

    # model
    model = tf.keras.models.Model(inputs=[IN_LR, IN_HR], outputs=[OUT], name='UNET-DENSE')

    return model
#%% Extinct Versions

def UNET_o(input_shape, ups_factors, layer_N=[64, 96, 128, 160], mode='lrhr', input_stack_num=2, pool=True, activation='relu'):
    '''
    UNet with three down- and upsampling levels.

    Input
    ----------
        layer_N: Number of convolution filters on each down- and upsampling levels
                 Should be an iterable of four int numbers, e.g., [32, 64, 128, 256]

        input_shape: a tuple that feed input keras.layers.Input, e.g. (None, None, 3)

        input_stack_num: number of stacked convolutional layers before entering the first downsampling block

        pool: True for maxpooling, False for strided convolutional layers
        activation: 'relu' for ReLU or 'leaky' for Leaky ReLU

    Output
    ----------
        model: the Keras functional API model, i.e., keras.models.Model(inputs=[IN], outputs=[OUT])

    '''

    # input layer
    # input layer
    if mode=='lrhr':
        IN = x = tf.keras.layers.Input(input_shape, name='unet_in')
        for _, ups_size in enumerate(ups_factors):
            x = tf.keras.layers.UpSampling2D(size=ups_size, interpolation='bilinear')(x)
        
        X_en1 = CONV_stack(x, layer_N[0], kernel_size=3, stack_num=input_stack_num, activation=activation, name='unet_left0')
    
    elif mode=='hrhr':
        IN = tf.keras.layers.Input(input_shape, name='unet_in')
        X_en1 = CONV_stack(IN, layer_N[0], kernel_size=3, stack_num=input_stack_num, activation=activation, name='unet_left0')
        
    else:
        mode_options = "', '".join(["lrhr", "hrhr"])
        raise ValueError(f"Invalid upsampling method. Available options are '{mode_options}'.")
    
    # downsampling levels
    X_en2 = UNET_left(X_en1, layer_N[1], pool=pool, activation=activation, name='unet_left1')
    X_en3 = UNET_left(X_en2, layer_N[2], pool=pool, activation=activation, name='unet_left2')
    X4 = UNET_left(X_en3, layer_N[3], pool=pool, activation=activation, name='unet_bottom')

    # upsampling levels
    X_de3 = UNET_right(X4, X_en3, layer_N[2], activation=activation, name='unet_right2')
    X_de2 = UNET_right(X_de3, X_en2, layer_N[1], activation=activation, name='unet_right1')
    X_de1 = UNET_right(X_de2, X_en1, layer_N[0], activation=activation, name='unet_right0')

    # output
    OUT = CONV_stack(X_de1, 2, kernel_size=3, stack_num=1, activation=activation, name='unet_out')
    OUT = tf.keras.layers.Conv2D(1, 1, activation=tf.keras.activations.linear, padding='same', name='unet_exit')(OUT)

    # model
    model = tf.keras.models.Model(inputs=[IN], outputs=[OUT], name='UNet')

    return model

## NEST_UNET_o

def NEST_UNET_o(input_shape, ups_factors, layer_N=[64, 96, 128, 160], mode='lrhr', input_stack_num=2, pool=True, activation='relu'):
    '''
    Nest-UNet (or UNet++) with three down- and upsampling levels.

    Input
    ----------
        layer_N: Number of convolution filters on each down- and upsampling levels
                 Should be an iterable of four int numbers, e.g., [32, 64, 128, 256]

        input_shape: a tuple that feed input keras.layers.Input, e.g. (None, None, 3)
        input_stack_num: number of stacked convolutional layers before entering the first downsampling block
        activation: 'relu' for ReLU or 'leaky' for Leaky ReLU
        pool: True for maxpooling, False for strided convolutional layers

    Output
    ----------
        model: the Keras functional API model, i.e., keras.models.Model(inputs=[IN], outputs=[OUT])

    '''

    # input layer
    if mode=='lrhr':
        IN = x = tf.keras.layers.Input(input_shape, name='unet_in')
        for _, ups_size in enumerate(ups_factors):
            x = tf.keras.layers.UpSampling2D(size=ups_size, interpolation='bilinear')(x)
        
        X11_conv = CONV_stack(x, layer_N[0], kernel_size=3, stack_num=input_stack_num, activation=activation, name='unet_left0')
    
    elif mode=='hrhr':
        IN = tf.keras.layers.Input(input_shape, name='unet_in')
        X11_conv = CONV_stack(IN, layer_N[0], kernel_size=3, stack_num=input_stack_num, activation=activation, name='unet_left0')
        
    else:
        mode_options = "', '".join(["lrhr", "hrhr"])
        raise ValueError(f"Invalid upsampling method. Available options are '{mode_options}'.")

    # downsampling levels (same as in the UNET)
    X21_conv = UNET_left(X11_conv, layer_N[1], pool=pool, activation=activation, name='unet_left1')
    X31_conv = UNET_left(X21_conv, layer_N[2], pool=pool, activation=activation, name='unet_left2')
    X41_conv = UNET_left(X31_conv, layer_N[3], pool=pool, activation=activation, name='unet_bottom')

    # up-sampling part 1
    X12_conv = XNET_right(X21_conv, [X11_conv], layer_N[0], activation=activation, name='xnet_12')
    X22_conv = XNET_right(X31_conv, [X21_conv], layer_N[1], activation=activation, name='xnet_22')
    X32_conv = XNET_right(X41_conv, [X31_conv], layer_N[2], activation=activation, name='xnet_32')

    # up-sampling part 2
    X13_conv = XNET_right(X22_conv, [X11_conv, X12_conv], layer_N[0], activation=activation, name='xnet_right13')
    X23_conv = XNET_right(X32_conv, [X21_conv, X22_conv], layer_N[1], activation=activation, name='xnet_right23')

    # up-sampling part 3
    X14_conv = XNET_right(X23_conv, [X11_conv, X12_conv, X13_conv], layer_N[0], activation=activation, name='xnet_right14')

    # output
    OUT = CONV_stack(X14_conv, 2, kernel_size=3, stack_num=1, activation=activation, name='xnet_out')
    OUT = tf.keras.layers.Conv2D(1, 1, activation=tf.keras.activations.linear)(OUT)

    # model
    model = tf.keras.models.Model(inputs=[IN], outputs=[OUT], name='Nest-UNet')

    return model
#%% Network Construct Utilities

def input_dense(x, dropout=0.5):
    """
    Apply a dense layer to the input.

    Args:
        x (tf.Tensor): Input tensor.
        dropout (float): Dropout rate.

    Returns:
        tf.Tensor: Output tensor.

    """
    flatten = tf.keras.layers.Flatten()(x)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(dropout)(flatten)
    else:
        x = flatten
    return x

def conv_layer(x, n_filters=32, activation='prelu', padding='same', kernel_size=(2, 3, 3),
              pooling=True, bn=True, strides=1):
    """
    Create a convolutional layer.
    """
    if activation=='prelu':
        x = tf.keras.layers.Conv2D(filters=n_filters, padding=padding, kernel_size=kernel_size, strides=strides)(x)
        x = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(x)
    else:
        x = tf.keras.layers.Conv2D(filters=n_filters, activation=activation, padding=padding, kernel_size=kernel_size, strides=strides)(x)
    if pooling:
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    return x

def contruct_base_conv(x, layer_filters=[50, 32, 16], bn=True, padding='same', kernel_size=3, pooling=True,
                        activation='relu', dropout=0.5, strides=1):
    """
    Construct base convolutional layers for the model.
    """
    for layer_filt in layer_filters:
        x = conv_layer(x, n_filters=layer_filt, bn=bn, padding=padding, kernel_size=kernel_size, pooling=pooling,
                       activation=activation, strides=strides)
    flatten = tf.keras.layers.Flatten()(x)
    if dropout > 0.0:
        flatten = tf.keras.layers.Dropout(dropout)(flatten)
    return flatten




# SRDRN Activation block
def SRDRN_activation(model, activation='prelu'):    
    """
    Apply activation function to the given model.
    """
    
    if activation == 'relu':
        return tf.keras.layers.ReLU()(model)
    elif activation == 'leaky':
        return tf.keras.layers.LeakyReLU(alpha=0.3)(model)
    elif activation == 'prelu':    
        return tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    else:
        return model  # Return the input model if no activation is specified

# SRDRN Residual block
def SRDRN_residual_block(model, kernal_size, n_filters, strides, regularizer, initializer, activation):
    """
    Constructs a residual block for the Super-resolution Residual Deep Neural Network (SRDRN).
    """
    gen = model    
    model = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = kernal_size, strides = strides, padding = "same", kernel_regularizer=regularizer, kernel_initializer=initializer)(model)
    model = tf.keras.layers.BatchNormalization(momentum = 0.5)(model)
    model = SRDRN_activation(model, activation=activation)
    model = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = kernal_size, strides = strides, padding = "same", kernel_regularizer=regularizer, kernel_initializer=initializer)(model)
    model = tf.keras.layers.BatchNormalization(momentum = 0.5)(model)    
    model = tf.keras.layers.add([gen, model])    
    return model

# SRDRN Upsampling block
def SRDRN_upsampling_block(model, ups_size, n_filters, activation, regularizer, initializer, interpolation):
    """
    Constructs an upsampling block for the Super-resolution Residual Deep Neural Network (SRDRN).
    """
    model = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = 3, strides = 1, padding = "same", kernel_regularizer=regularizer, kernel_initializer=initializer)(model)
    model = tf.keras.layers.UpSampling2D(size = ups_size, interpolation = interpolation)(model)
    model = SRDRN_activation(model, activation=activation)
    # print('*'*20, str(interpolation), '*'*20)
    return model

def SRDRN_convtranspose_block(model, ups_size, n_filters, activation, regularizer, initializer):
    """
    Constructs an upsampling block for the Super-resolution Residual Deep Neural Network (SRDRN).
    """
    model = tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=3, strides=ups_size, padding="same",
                                         kernel_regularizer=regularizer, kernel_initializer=initializer)(model)
    model = SRDRN_activation(model, activation=activation)
    return model

def stride_conv(X, channel, pool_size=2, activation='relu', name='X'):
    '''
    stride convolutional layer --> batch normalization --> Activation
    *Proposed to replace max- and average-pooling layers

    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        pool_size: size of stride
        activation: 'relu' for ReLU or 'leaky' for Leaky ReLU
        name: name of created keras layers

    Output
    ----------
        X: output tensor
    '''
    # linear convolution with strides
    X = tf.keras.layers.Conv2D(channel, pool_size, strides=(pool_size, pool_size), padding='valid',
                            use_bias=False, kernel_initializer='he_normal', name=name+'_stride_conv')(X)
    # batch normalization
    X = tf.keras.layers.BatchNormalization(axis=3, name=name+'_stride_conv_bn')(X)

    # activation
    if activation == 'relu':
        X = tf.keras.layers.ReLU(name=name+'_stride_conv_relu')(X)
    elif activation == 'leaky':
        X = tf.keras.layers.LeakyReLU(alpha=0.3, name=name+'_stride_conv_leaky')(X)
    elif activation == 'prelu':
            X = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2], name=name+'_stride_conv_prelu')(X)

    return X

def CONV_stack(X, channel, kernel_size=3, stack_num=2, activation='relu', name='conv_stac'):
    '''
    (Convolutional layer --> batch normalization --> Activation)*stack_num

    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        stack_num: number of stacked convolutional layers
        activation: 'relu' for ReLU or 'leaky' for Leaky ReLU
        name: name of created keras layers

    Output
    ----------
        X: output tensor

    '''

    # stacking Convolutional layers
    for i in range(stack_num):

        # linear convolution
        X = tf.keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False,
                                kernel_initializer='he_normal', name=name+'_stack{}_conv'.format(i))(X)

        # batch normalization
        X = tf.keras.layers.BatchNormalization(axis=3, name=name+'_stack{}_bn'.format(i))(X)

        # activation
        if activation == 'relu':
            X = tf.keras.layers.ReLU(name=name+'_stack{}_relu'.format(i))(X)
        elif activation == 'leaky':
            X = tf.keras.layers.LeakyReLU(alpha=0.3, name=name+'_stack{}_leaky'.format(i))(X)
        elif activation == 'prelu':
            X = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2], name=name+'_stack{}_prelu'.format(i))(X)

    return X


def UNET_left(X, channel, kernel_size=3, pool_size=2, pool=True, activation='relu', name='left0'):
    '''
    Encoder block of UNet

    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        pool_size: size of stride
        pool: True for maxpooling, False for strided convolutional layers
        activation: 'relu' for ReLU or 'leaky' for Leaky ReLU
        name: name of created keras layers

    Output
    ----------
        X: output tensor

    '''

    # maxpooling layer vs strided convolutional layers
    if pool:
        X = tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='{}_pool'.format(name))(X)
    else:
        X = stride_conv(X, channel, pool_size, activation=activation, name=name)

    # stack linear convolutional layers
    X = CONV_stack(X, channel, kernel_size, stack_num=1, activation=activation, name=name)

    return X

def UNET_right(X, X_left, channel, kernel_size=3, pool_size=2, activation='relu', name='right0'):
    '''
    Decoder block of UNet

    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        pool_size: size of transpose stride, expected to be the same as "UNET_left"
        activation: 'relu' for ReLU or 'leaky' for Leaky ReLU
        name: name of created keras layers

    Output
    ----------
        X: output tensor

    '''

    # Transpose convolutional layer --> stacked linear convolutional layers
    X = tf.keras.layers.Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size),
                                     padding='same', name=name+'_trans_conv')(X)
    X = CONV_stack(X, channel, kernel_size, stack_num=1, activation=activation, name=name+'_conv_after_trans')

    # Tensor concatenation
    H = tf.keras.layers.concatenate([X_left, X], axis=3)

    # stacked linear convolutional layers after concatenation
    H = CONV_stack(H, channel, kernel_size, stack_num=1, activation=activation, name=name+'_conv_after_concat')

    return H

def XNET_right(X_conv, X_list, channel, kernel_size=3, pool_size=2, activation='relu', name='right0'):
    '''
    Decoder block of Nest-UNet

    Input
    ----------
        X: input tensor
        X_list: a list of other corresponded input tensors (see Sha 2020b, Figure 2)
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        pool_size: size of transpose stride, expected to be the same as "UNET_left"
        activation: 'relu' for ReLU or 'leaky' for Leaky ReLU
        name: name of created keras layers

    Output
    ----------
        X: output tensor

    '''

    # Transpose convolutional layer --> concatenation
    X_unpool = tf.keras.layers.Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size),
                                            padding='same', name=name+'_trans_conv')(X_conv)

    # <--- *stacked convolutional can be applied here
    X_unpool = tf.keras.layers.concatenate([X_unpool]+X_list, axis=3, name=name+'_nest')

    # Stacked convolutions after concatenation
    X_conv = CONV_stack(X_unpool, channel, kernel_size, stack_num=2, activation=activation, name=name+'_conv_after_concat')

    return X_conv








