import matplotlib.pyplot as plt
import re
import numpy as np

# Path to your log file
log_file = "slurm_outputs/train-721809.out"  
output_plot = "data_loading_time_binned_avg.png"  # Output file name

# Lists to store extracted data
iterations = []
data_loading_times = []

# Regular expression to find "Step X: Data loading time: Y sec"
pattern = re.compile(r"Step (\d+): Data loading time: ([\d\.]+) sec")

# Read the log file and extract data
with open(log_file, "r") as file:
    for line in file:
        match = pattern.search(line)
        if match:
            iteration = int(match.group(1))  # Extract step number
            loading_time = float(match.group(2))  # Extract loading time
            iterations.append(iteration)
            data_loading_times.append(loading_time)

# Define window size (every 50 steps)
window_size = 100

# Compute the mean for non-overlapping windows
binned_iterations = []
binned_averages = []

for i in range(0, len(data_loading_times), window_size):
    if i + window_size <= len(data_loading_times):  # Ensure full window
        avg_loading_time = np.mean(data_loading_times[i:i+window_size])
        binned_averages.append(avg_loading_time)
        binned_iterations.append(iterations[i + window_size - 1])  # Use last step of the window for x-axis

# Plot the binned averages
plt.figure(figsize=(10, 5))
plt.plot(binned_iterations, binned_averages, marker="o", linestyle="-", color="b", markersize=5)

# Labels and title
plt.xlabel("Iteration")
plt.ylabel("Mean Data Loading Time (seconds)")
plt.title(f"Data Loading Time Average (Every {window_size} Steps)")
plt.grid(True)

# Save the figure instead of showing it
plt.savefig(output_plot, dpi=300, bbox_inches="tight")
print(f"Plot saved as {output_plot}")
