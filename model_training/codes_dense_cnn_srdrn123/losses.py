#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:15:55 2024

@author: midhunm
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
# import tensorflow.keras.backend as K

#%% Define custom loss functions

def mae_loss_with_mass_conservation(y_true, y_pred, lambda_conserv=1e-4):
    # Calculate the Mean Absolute Error (MAE)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Calculate the mass conservation term
    mass_conservation_error = tf.abs(tf.reduce_sum(y_true) - tf.reduce_sum(y_pred))
    
    # Combine MAE and the mass conservation term
    total_loss = mae + lambda_conserv * mass_conservation_error
    return total_loss

#%%

def weighted_mse(y_true, y_pred, clip_min=np.log10(0.1 + 1), clip_max=np.log10(100 + 1)):
    """
    Weighted mean squared error (MSE) loss function.
    Weights are calculated based on the true values, with higher weights assigned to larger values.
    """
    # Calculate the weights for each data point
    weights = tf.clip_by_value(y_true, clip_min, clip_max)
    # Calculate the weighted mean squared error
    loss = K.mean(tf.multiply(weights, tf.square(tf.subtract(y_pred, y_true))))
    return loss

def weighted_mae(y_true, y_pred, clip_min=np.log10(0.1 + 1), clip_max=np.log10(100 + 1)):
    """
    Weighted mean absolute error (MAE) loss function.
    Weights are calculated based on the true values, with higher weights assigned to larger values.
    """
    # Calculate the weights for each data point
    weights = tf.clip_by_value(y_true, clip_min, clip_max)
    # Calculate the weighted mean absolute error
    loss = K.mean(tf.multiply(weights, tf.abs(y_true - y_pred)))
    return loss

#%%



def gamma_mse_metric(y_true, y_pred, thres=0.5):
    """
    Custom metric for mean squared error of gamma distribution parameters.

    Args:
        y_true (tensor): True target values.
        y_pred (tensor): Predicted values, including shape parameter, scale parameter, and occurrence probability.
        thres (float): Threshold for rainfall occurrence.

    Returns:
        tensor: The calculated mean squared error.

    """
    # Extract predicted values
    occurence = y_pred[:,:,:,-1]
    shape_param = K.exp(y_pred[:,:,:,0])
    scale_param = K.exp(y_pred[:,:,:,1])
    
    # Calculate the rainfall using the gamma distribution
    rainfall = shape_param * scale_param * tf.cast(occurence > thres, 'float32')
    
    # Calculate mean squared error between predicted and true rainfall
    return tf.keras.losses.mean_squared_error(rainfall, y_true)

def gamma_loss(y_true, y_pred, eps=3e-2):
    """
    Custom loss function for a gamma distribution parameterization.

    Args:
        y_true (tensor): True target values.
        y_pred (tensor): Predicted values, including shape parameter, scale parameter, and occurrence probability.
        eps (float): Small constant to prevent numerical instability.

    Returns:
        tensor: The calculated loss value.

    """
    # Extract predicted values
    
    y_true = y_true[:,:,:,0]
    
    occurence = y_pred[:,:,:,-1]
    shape_param = K.exp(y_pred[:,:,:,0])
    scale_param = K.exp(y_pred[:,:,:,1])
    
    # Convert y_true to a binary indicator for rain (1 if > 0.0, 0 otherwise)
    bool_rain = tf.cast(y_true > 0.0, 'float32')
    eps = tf.cast(eps, 'float32')
    
    # Calculate the gamma loss
    loss1 = ((1 - bool_rain) * tf.math.log(1 - occurence + eps) +
             bool_rain * (K.log(occurence + eps) +
                         (shape_param - 1) * K.log(y_true + eps) -
                         shape_param * tf.math.log(scale_param + eps) -
                         tf.math.lgamma(shape_param) -
                         y_true / (scale_param + eps)))
    
    # Calculate the absolute mean of the loss
    output_loss = tf.abs(K.mean(loss1))
    return output_loss

class BernoulliGammaLoss(Loss):
    def __init__(self, name="bernoulli_gamma_loss", reduction='sum_over_batch_size'):
        super().__init__(name=name, reduction=reduction)

    def call(self, y_true, y_pred, epsilon=1e-6):
        # Ensure epsilon is cast to the right type
        epsilon = tf.cast(epsilon, tf.float32)
        
        # Extract true values (precipitation) and predicted occurrence, shape, and scale parameters
        y_true = y_true[:,:,:,0]
        occurrence = y_pred[:,:,:,-1]
        shape_parameter = tf.exp(y_pred[:,:,:,0])
        scale_parameter = tf.exp(y_pred[:,:,:,1])

        # Boolean mask for non-zero precipitation
        bool_rain = tf.cast(y_true > 0.0, tf.float32)

        # Log-likelihood for Bernoulli-Gamma distribution
        log_likelihood = (
            (1 - bool_rain) * tf.math.log(1 - occurrence + epsilon) +
            bool_rain * (
                tf.math.log(occurrence + epsilon) +
                (shape_parameter - 1) * tf.math.log(y_true + epsilon) -
                shape_parameter * tf.math.log(scale_parameter + epsilon) -
                tf.math.lgamma(shape_parameter + epsilon) -
                y_true / (scale_parameter + epsilon)
            )
        )

        # Return the mean negative log-likelihood as the loss
        return -tf.reduce_mean(log_likelihood)


class CustomPhysicsLoss(Loss):
    def __init__(self, reg_loss='WMAE', alpha=1, beta=1e-4, clip_min=np.log10(0.1 + 1), clip_max=np.log10(100 + 1), name="custom_physics_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.reg_loss = reg_loss

    def __str__(self):
        return (
            f"CustomPhysicsLoss(reg_loss={self.reg_loss}, alpha={self.alpha}, beta={self.beta}, "
            f"clip_min={self.clip_min}, clip_max={self.clip_max})"
        )
    
    def call(self, y_true, y_pred):
        # Define a dictionary mapping the regular loss options to the loss functions
        loss_dict = {
            'MSE': tf.reduce_mean(tf.square(y_true - y_pred)),
            'MAE': tf.reduce_mean(tf.abs(y_true - y_pred)),
            'WMSE': weighted_mse(y_true, y_pred, clip_min=self.clip_min, clip_max=self.clip_max),
            'WMAE': weighted_mae(y_true, y_pred, clip_min=self.clip_min, clip_max=self.clip_max),
        }
        
        # Select the regular loss function based on the reg_loss parameter
        regular_loss = loss_dict.get(self.reg_loss, None)
        
        if regular_loss is None:
            raise ValueError("Invalid choice of regular loss! Available options are 'WMAE', 'MAE', 'WMSE', and 'MSE'")
    
        # Calculate the mass conservation regularization term
        mass_conservation_error = tf.reduce_mean(tf.abs(tf.reduce_sum(y_true, axis=(1,2)) - tf.reduce_sum(y_pred, axis=(1,2))))
    
        # Combine the average difference term and the mass conservation regularization term
        custom_physics_loss = self.alpha * regular_loss + self.beta * mass_conservation_error
    
        return custom_physics_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta,
            'clip_min': self.clip_min,
            'clip_max': self.clip_max,
            'reg_loss': self.reg_loss,
        })
        return config

#%%

# def gamma_loss_(y_true, y_pred, eps=3e-2):
#     """
#     Custom loss function for a gamma distribution parameterization.

#     Args:
#         y_true (tensor): True target values.
#         y_pred (tensor): Predicted values, including shape parameter, scale parameter, and occurrence probability.
#         eps (float): Small constant to prevent numerical instability.

#     Returns:
#         tensor: The calculated loss value.

#     """
#     # Extract predicted values
    
#     y_true = y_true[:,:,:,0]
    
#     occurence = y_pred[:,:,:,-1]
#     shape_param = K.exp(y_pred[:,:,:,0])
#     scale_param = K.exp(y_pred[:,:,:,1])
    
#     # Convert y_true to a binary indicator for rain (1 if > 0.0, 0 otherwise)
#     bool_rain = tf.cast(y_true > 0.0, 'float32')
#     eps = tf.cast(eps, 'float32')
    
#     # Calculate the gamma loss
#     loss1 = ((1 - bool_rain) * tf.math.log(1 - occurence + eps) +
#              bool_rain * (K.log(occurence + eps) +
#                          (shape_param - 1) * K.log(y_true + eps) -
#                          shape_param * tf.math.log(scale_param + eps) -
#                          tf.math.lgamma(shape_param) -
#                          y_true / (scale_param + eps)))
    
#     # Calculate the absolute mean of the loss
#     output_loss = tf.abs(K.mean(loss1))
#     return output_loss

# def gamma_loss_1d_(y_true, y_pred, eps=3e-2):
#     """
#     Custom loss function for a gamma distribution parameterization.

#     Args:
#         y_true (tensor): True target values.
#         y_pred (tensor): Predicted values, including shape parameter, scale parameter, and occurrence probability.
#         eps (float): Small constant to prevent numerical instability.

#     Returns:
#         tensor: The calculated loss value.

#     """
#     # Extract predicted values
#     occurence = y_pred[:,-1]
#     shape_param = K.exp(y_pred[:,0])
#     scale_param = K.exp(y_pred[:,1])
    
#     # Convert y_true to a binary indicator for rain (1 if > 0.0, 0 otherwise)
#     bool_rain = tf.cast(y_true > 0.0, 'float32')
#     eps = tf.cast(eps, 'float32')
    
#     # Calculate the gamma loss
#     loss1 = ((1 - bool_rain) * tf.math.log(1 - occurence + eps) +
#              bool_rain * (K.log(occurence + eps) +
#                          (shape_param - 1) * K.log(y_true + eps) -
#                          shape_param * tf.math.log(scale_param + eps) -
#                          tf.math.lgamma(shape_param) -
#                          y_true / (scale_param + eps)))
    
#     # Calculate the absolute mean of the loss
#     output_loss = tf.abs(K.mean(loss1))
#     return output_loss