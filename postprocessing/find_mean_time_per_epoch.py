import os
import pandas as pd

# Specify the directory containing the CSV files
directory = "/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/RAWRUNS/postprocs/timeperepoch"

# Prepare an empty list to store filename and min_val_loss
results = []

# Iterate over each file in the directory
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".csv"):# and "p07a" in filename: # Edit here
        file_path = os.path.join(directory, filename)

        # Load the CSV file without headers
        df = pd.read_csv(file_path, header=None)
        
        # Exclude the first value and compute the mean
        mean_value = df.iloc[1:, 0].mean() 
            
        # Append the result to the list
        results.append([filename, mean_value])

# Create a DataFrame from the results
results_df = pd.DataFrame(results, columns=['filename', 'timeperepoch'])

# Sort the DataFrame by filename in alphabetical order
results_df = results_df.sort_values(by='filename', ascending=True)

# Save the DataFrame to a new CSV file
output_path = "mean_time_per_epoch.csv"
results_df.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")
