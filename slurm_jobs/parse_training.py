import re
import numpy as np

# Path to your txt file
filename = 'slurm-9624952.out'

# Open and read the file
with open(filename, 'r') as f:
    text = f.read()

# Split into runs
runs = text.split("Loading drug dataset...")[1:]  # Skip the first split

# Storage
max_test_acc = []
max_test_f1 = []
train_times = []

# Regular expressions
metrics_pattern = re.compile(r"\[\d+,\d+\] \| tst acc:(\d+\.\d+), f1:(\d+\.\d+)")
train_time_pattern = re.compile(r"Total training time: ([\d\.]+) GPU seconds")

# Process each run
for run in runs:
    metrics = metrics_pattern.findall(run)
    if metrics:
        # Find max F1 and corresponding accuracy
        max_f1 = -1.0
        best_acc = -1.0
        for acc, f1 in metrics:
            f1 = float(f1)
            acc = float(acc)
            if f1 > max_f1:
                max_f1 = f1
                best_acc = acc
        max_test_f1.append(max_f1)
        max_test_acc.append(best_acc)
    
    time_match = train_time_pattern.search(run)
    if time_match:
        train_times.append(float(time_match.group(1)))

# Convert to numpy arrays
max_test_acc = np.array(max_test_acc)
max_test_f1 = np.array(max_test_f1)
train_times = np.array(train_times)

# Print mean and sample std
print(f"Max Test Accuracy (at max F1): Mean = {max_test_acc.mean():.2f} +/- {max_test_acc.std(ddof=1):.2f}")
print(f"Max Test F1 Score: Mean = {max_test_f1.mean():.2f} +/- {max_test_f1.std(ddof=1):.2f}")
print(f"Total Training Time: Mean = {train_times.mean():.2f} +/- {train_times.std(ddof=1):.2f}")