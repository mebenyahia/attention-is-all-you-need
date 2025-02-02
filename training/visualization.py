import os

import pandas as pd

from visualization.plotting import plot_metrics

# Define the file path
file_path = 'training_log.csv'
os.makedirs('results/figures', exist_ok=True)

# Read the CSV file
df = pd.read_csv(file_path, delimiter=';')

# Extract the training and validation losses
train_losses = df['Train_Loss'].tolist()
val_losses = df['Val_Loss'].tolist()
sum_time = sum(df['Time'].tolist())
print(sum_time)

# Plot the metrics
plot_metrics(train_losses, val_losses)