import re
from statistics import mean, stdev
from collections import defaultdict

# Load the full input file
with open("slurm-9556659.out", "r") as f:
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

# Compute averages and std devs
summary = []
for alpha, metrics in sorted(results.items()):
    avg_acc = mean(metrics["acc"])
    avg_f1 = mean(metrics["f1"])
    avg_time = mean(metrics["time"])
    std_acc = stdev(metrics["acc"]) if len(metrics["acc"]) > 1 else 0.0
    std_f1 = stdev(metrics["f1"]) if len(metrics["f1"]) > 1 else 0.0
    std_time = stdev(metrics["time"]) if len(metrics["time"]) > 1 else 0.0
    summary.append((
        alpha,
        round(avg_acc, 2), round(std_acc, 2),
        round(avg_f1, 2), round(std_f1, 2),
        round(avg_time, 2), round(std_time, 2)
    ))

# Print results
for row in summary:
    print(f"alpha={row[0]:.3f} | acc: {row[1]:.2f}+/-{row[2]:.2f} | f1: {row[3]:.2f}±{row[4]:.2f} | time: {row[5]:.2f}±{row[6]:.2f}")

# Optional: Pandas display
# import pandas as pd
# import ace_tools as tools
# df = pd.DataFrame(summary, columns=[
#     "alpha", "avg_test_acc", "std_test_acc",
#     "avg_test_f1", "std_test_f1",
#     "avg_training_time", "std_training_time"
# ])
# tools.display_dataframe_to_user(name="Alpha Hyperparameter Tuning Summary", dataframe=df)