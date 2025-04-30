import re
import numpy as np

# Path to your txt file
filename = 'auto.out'

# Open and read the file
with open(filename, 'r') as f:
    text = f.read()

# Split into runs
runs = text.split("Loading drug dataset...")[1:]  # Skip the first split

# Storage
last_test_acc = []
last_test_f1 = []
last_train_time = []

# Regular expressions
metrics_pattern = re.compile(r"\[\d+,\d+\] \| tst acc:(\d+\.\d+), f1:(\d+\.\d+)")
train_time_pattern = re.compile(r"Total training time: ([\d\.]+) GPU seconds")

# Process each run
for run in runs:
    metrics = metrics_pattern.findall(run)
    if metrics:
        acc, f1 = metrics[-1]  # Only take the LAST recorded [epoch,batch]
        last_test_acc.append(float(acc))
        last_test_f1.append(float(f1))
    
    time_match = train_time_pattern.search(run)
    if time_match:
        last_train_time.append(float(time_match.group(1)))

# Convert to numpy arrays
last_test_acc = np.array(last_test_acc)
last_test_f1 = np.array(last_test_f1)
last_train_time = np.array(last_train_time)

# Print mean and sample std
print(f"Final Test Accuracy: Mean = {last_test_acc.mean():.2f} +/- {last_test_acc.std(ddof=1):.2f}")
print(f"Final Test F1 Score: Mean = {last_test_f1.mean():.2f} +/- {last_test_f1.std(ddof=1):.2f}")
print(f"Total Training Time: Mean = {last_train_time.mean():.2f} +/- {last_train_time.std(ddof=1):.2f}")