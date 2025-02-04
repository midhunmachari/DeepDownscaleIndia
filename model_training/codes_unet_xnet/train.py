# Import libraries
import datetime
import numpy as np
import tensorflow as tf
from utils import TimeHistory, make_predictions, negtozero, r_logtrans

## The Model Training Class ##

class ModelTraining():
    
    def __init__(self,
                 prefix='test',
                 pwd = './'
                 ):
        
        self.prefix = prefix
        self.pwd = pwd
        self.best_model = None
    
    def train(self,
              model,
              train_data,
              valid_data,
              epochs=10, 
              loss='mae', 
              batch_size=64, 
              learning_rate=10**-4,   
              ):
        
        """To  train the compiled SRDRN/UNET/XNET/FSRCNN Model"""
        
        # -----------------------
        # Build tf.data pipeline
        # -----------------------
        
        buffer_len = len(train_data[0])
        
        if len(train_data)==3:
            train_dataset = tf.data.Dataset.from_tensor_slices(((train_data[0], train_data[1]), train_data[2]))
            valid_dataset = tf.data.Dataset.from_tensor_slices(((valid_data[0], valid_data[1]), valid_data[2]))
        elif len(train_data)==2:
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1]))
            valid_dataset = tf.data.Dataset.from_tensor_slices((valid_data[0], valid_data[1]))
        else:
            raise ValueError("Invalid number of Training inputs")
         
        # Create a dataset with the train data
        train_dataset = train_dataset.shuffle(buffer_size=buffer_len)
        train_dataset = train_dataset.batch(batch_size=batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        print("Train Dataset Element Spec:", train_dataset.element_spec)
        
        # Create a dataset with the validation data
        valid_dataset = valid_dataset.batch(batch_size=batch_size)
        valid_dataset = valid_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        print("Valid Dataset Element Spec:", valid_dataset.element_spec)
        
        # -------------------------
        # Define Horovod callbacks
        # -------------------------
        model_path = f'{self.pwd}/../{self.prefix}_model.keras' # model_e{{epoch:04d}}
        modelsave_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                                    save_weights_only=False,
                                                                    monitor='val_loss',
                                                                    mode='min',
                                                                    save_best_only=True)


        # -------------------------------------------------------
        # Define model callbacks (conditionally apply for GPU 0)
        # -------------------------------------------------------
        csv_logger = tf.keras.callbacks.CSVLogger(f"{self.pwd}/../{self.prefix}_logs.csv", append=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=12, min_lr=1e-8)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=32, restore_best_weights=True)
        timeperepoch = TimeHistory(f"{self.pwd}/../{self.prefix}_timeperepoch.csv")
        
        # -------------------------------------
        # Add TensorBoard callback
        # -------------------------------------
        log_dir = f'{self.pwd}/../{self.prefix}_logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                     histogram_freq=1, 
                                                    #  profile_batch = '0,20'
                                                     )
        
        # -------------------------------------
        # Compile the model and start training
        # -------------------------------------
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                      loss=loss,
                      metrics=[tf.keras.losses.MeanAbsoluteError(), 
                               tf.keras.losses.MeanSquaredError()
                               ]
                      )     
        
        # Print all details
        print('*'*100)
        print(f"Exp.: {self.prefix}")
        print(f"\nModel name: {model.name}")
        print(f"Training with the Loss fn.: {loss}")
        print(f"Learning rate set to {learning_rate}")
        print(f"Batchsize set to {batch_size}")
        print(f"Training scheduled for {epochs} epochs.")
        print('*'*100)
        print("...MODEL TRAINING STARTS NOW...")
        
        model.fit(train_dataset,
                  epochs=epochs, 
                  verbose=2,
                  validation_data=valid_dataset, 
                  callbacks=[modelsave_callback, 
                             csv_logger,
                             reduce_lr, 
                             early_stop, 
                             tensorboard,
                             timeperepoch,
                             ]
                  )    

    def load_best_model(self):
        if self.best_model is None:  # Only load the model once
            print(f"\nLoading model from {self.prefix}_model.keras...")
            self.best_model = tf.keras.models.load_model(f'{self.pwd}/../{self.prefix}_model.keras', compile=False)

    def gen_predictions(self, X_test, loss_fn=None, batch_size=32):
        """Generates predictions and optionally saves them."""
        self.load_best_model()
        
        print(f"\nGenerating test data for {self.prefix}...")
        
        if len(X_test)==1:
            print(f"X_test shape: {X_test[0].shape}")
        else:
            print(f"X_test shape: {X_test[0].shape}")
            print(f"S_test shape: {X_test[1].shape}")
        
        # Generate predictions in batches
        gen_data = make_predictions(self.best_model, X_test, batch_size=batch_size, loss_fn=loss_fn, thres=0.5)
        
        # Optional: save the predictions, only if required
        output_path = f'{self.pwd}/../{self.prefix}_out_raw.npy'
        np.save(output_path, gen_data)
        print(f"Saved raw predictions to {output_path}")
        
        return gen_data

    def evaluations(self, X_test, y_test, loss_fn='gamma'):
        """Evaluate the model on the test set."""
        y_pred_raw = self.gen_predictions(X_test, loss_fn=loss_fn)
        
        # Transform predictions before evaluation
        y_pred = r_logtrans(negtozero(y_pred_raw))
        
        metrics = self.compute_metrics(y_pred, y_test[:,:,:,0])
        
        return metrics

    def compute_metrics(self, y_pred, y_true):
        """Example evaluation metric computation function."""
        # Placeholder for custom evaluation logic
        mse = np.mean((y_pred - y_true)**2)  # Example: Mean Squared Error
        return {'mse': mse}

###############################################################################################################
