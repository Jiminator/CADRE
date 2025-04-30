import re
from statistics import mean
from collections import defaultdict

# Load the full input file
with open("tune_alpha.out", "r") as f:
    lines = f.readlines()

# Initialize storage
results = defaultdict(lambda: {"acc": [], "f1": [], "time": []})
current_alpha = None

# Regular expressions
alpha_re = re.compile(r"alpha\s*=\s*([0-9.]+)")
final_metrics_re = re.compile(r"\[\d+,\d+\] \| tst acc:(\d+\.\d+), f1:(\d+\.\d+), auc:")
total_time_re = re.compile(r"Total training time:\s+([0-9.]+) GPU seconds")

for line in lines:
    # Check for alpha value
    alpha_match = alpha_re.search(line)
    if alpha_match:
        current_alpha = float(alpha_match.group(1))

    # Extract test acc and f1 score from final result
    final_match = final_metrics_re.search(line)
    if final_match and current_alpha is not None:
        acc = float(final_match.group(1))
        f1 = float(final_match.group(2))
        results[current_alpha]["acc"].append(acc)
        results[current_alpha]["f1"].append(f1)

    # Extract total training time
    time_match = total_time_re.search(line)
    if time_match and current_alpha is not None:
        total_time = float(time_match.group(1))
        results[current_alpha]["time"].append(total_time)

# Compute averages
summary = []
for alpha, metrics in sorted(results.items()):
    avg_acc = mean(metrics["acc"])
    avg_f1 = mean(metrics["f1"])
    avg_time = mean(metrics["time"])
    summary.append((alpha, round(avg_acc, 2), round(avg_f1, 2), round(avg_time, 2)))
print(summary)
# import pandas as pd
# import ace_tools as tools; tools.display_dataframe_to_user(name="Alpha Hyperparameter Tuning Summary", dataframe=pd.DataFrame(summary, columns=["alpha", "avg_test_acc", "avg_test_f1", "avg_training_time (s)"]))