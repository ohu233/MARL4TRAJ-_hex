import pandas as pd

# Read the data
data = pd.read_csv('SAC_5000_eps_inrealmap_44_training_results.csv')

# Calculate IQR for the 'Return' column
Q1 = data['Return'].quantile(0.25)  # First quartile
Q3 = data['Return'].quantile(0.75)  # Third quartile
IQR = Q3 - Q1  # Interquartile range

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = (data['Return'] < lower_bound) | (data['Return'] > upper_bound)

# Replace outliers with the median of the 'Return' column
median_value = data['Return'].median()
data.loc[outliers, 'Return'] = median_value

# Save the cleaned data (optional)
data.to_csv('cleaned_data.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set global font sizes
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,          # Base font size
    'axes.labelsize': 20,     # Axis label font size
    'axes.titlesize': 24,     # Title font size
    'xtick.labelsize': 18,    # X-axis tick font size
    'ytick.labelsize': 18,    # Y-axis tick font size
    'legend.fontsize': 16,    # Legend font size
    'figure.titlesize': 20,   # Figure title font size
})

# Read data
data = pd.read_csv('cleaned_data.csv')

# Compute moving statistics (window size 100)
window_size = 300

# Reward statistics
data['Moving Average Reward'] = data['Return'].rolling(window=window_size, min_periods=1).mean()
data['Reward Std Dev'] = data['Return'].rolling(window=window_size, min_periods=1).std()

# Loss statistics
data['Moving Average Q-Loss'] = data['Loss'].rolling(window=window_size, min_periods=1).mean()
data['Q-Loss Std Dev'] = data['Loss'].rolling(window=window_size, min_periods=1).std()

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Configure grid lines
ax1.grid(True, linestyle='--', alpha=0.6)
ax2.grid(True, linestyle='--', alpha=0.6)

# Plot reward (left)
ax1.plot(data['Episode'], data['Return'], color='lightblue', alpha=0.3, label='Reward')
ax1.plot(data['Episode'], data['Moving Average Reward'], color='red', label='Moving Average Reward')
ax1.fill_between(
    data['Episode'],
    data['Moving Average Reward'] - data['Reward Std Dev'],
    data['Moving Average Reward'] + data['Reward Std Dev'],
    color='blue',
    alpha=0.2,
    label='Reward ± Std Dev'
)
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Reward')
ax1.set_title('Reward of the mode identification agent')
ax1.set_ylim(-0.5, 3)
ax1.legend(loc='lower right')

# Plot loss (right)
ax2.plot(data['Episode'], data['Loss'], color='green', alpha=0.3, label='Q-Loss')
ax2.plot(data['Episode'], data['Moving Average Q-Loss'], color='darkgreen', linewidth=2, label='Moving Average Loss')
ax2.fill_between(
    data['Episode'],
    data['Moving Average Q-Loss'] - data['Q-Loss Std Dev'],
    data['Moving Average Q-Loss'] + data['Q-Loss Std Dev'],
    color='green',
    alpha=0.15,
    label='Loss ± Std Dev'
)
ax2.set_xlabel('Episodes')
ax2.set_ylabel('Loss')
ax2.set_title('Q-Loss of the mode identification agent')
ax2.set_ylim(-0.0025, 0.0150)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()