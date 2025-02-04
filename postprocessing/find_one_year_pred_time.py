import time
import os
import numpy as np
import tensorflow as tf
import pandas as pd

# Model directory and list of models
directory = "/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/RAWRUNS/postprocs/sel_models/u0"

# Prepare an output CSV file with headers
output_file = "time_taken_for_one_year_prediction_u0.csv"
output_df = pd.DataFrame(columns=["MODEL_NAME", "ONE_YEAR_PRED_TIME", "TOTAL_PARAMETERS"])
output_df.to_csv(output_file, index=False)  # Save headers

# Loop through each model and measure prediction time
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".keras"):
        file_path = os.path.join(directory, filename)
        print(filename)
    # Load the model
    model = tf.keras.models.load_model(f'{directory}/{filename}', compile=False)

    # Print model summary
    print(model.summary())

    # Total number of parameters in the model
    total_params = model.count_params()

    # Create dummy data
    if 'u0' in filename and ('i01' in filename or 'i03' in filename):
        inp1 = np.random.rand(365, *model.input_shape[0][1:]).astype(np.float32)
        inp2 = np.random.rand(365, *model.input_shape[1][1:]).astype(np.float32)
        dummy_data = [inp1, inp2]
    
    else:
        dummy_data = np.random.rand(365, *model.input_shape[1:]).astype(np.float32)

    # Start the timer
    start_time = time.time()

    # Make predictions
    predictions = model.predict(dummy_data)

    # Stop the timer
    end_time = time.time()

    # Calculate the time taken
    time_taken = end_time - start_time

    # Append the results to the CSV file
    results = pd.DataFrame([{"MODEL_NAME": filename, "ONE_YEAR_PRED_TIME": time_taken, "TOTAL_PARAMETERS": total_params}])
    results = results.sort_values(by='MODEL_NAME', ascending=True)
    results.to_csv(output_file, mode='a', header=False, index=False)

    print(f"Time taken for one year prediction: {time_taken} seconds")
    print(f"Total parameters in model {filename}: {total_params}")