import os
import pandas as pd

# Specify the directory containing the CSV files
directory = "/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/RAWRUNS/postprocs/trainlogs"

# Prepare an empty list to store filename and min_val_loss
results = []

# Iterate over each file in the directory
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".csv") and "p07a" in filename: # Edit here
        file_path = os.path.join(directory, filename)
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Check if 'val_loss' column exists in the file
        if 'val_loss' in df.columns:
            # Find the index (0-based) and value of the minimum 'val_loss'
            min_index = df['val_loss'].idxmin() + 1  # 1-based index
            min_val_loss = df['val_loss'].min()
            
            # Append the result to the list
            results.append([filename, min_val_loss, min_index])

# Create a DataFrame from the results
results_df = pd.DataFrame(results, columns=['filename', 'min_val_loss', 'index'])

# Sort the DataFrame by filename in alphabetical order
results_df = results_df.sort_values(by='filename', ascending=True)

# Save the DataFrame to a new CSV file
output_path = "epochs_till_convergence_summary.csv"
results_df.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")
