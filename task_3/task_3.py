import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. Parse the log file and extract data
log_file = 'a_migineishvili25_93254_server.log'
pattern = r'\[(.*?)\]'
req_counts = {}

# Log time format: 2024-03-22 18:01:24+04:00
time_format = "%Y-%m-%d %H:%M:%S%z"

print("Reading and processing log file...")
with open(log_file, 'r') as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            time_str = match.group(1)
            dt = datetime.strptime(time_str, time_format)
            ts = int(dt.timestamp()) # Convert to Unix timestamp
            req_counts[ts] = req_counts.get(ts, 0) + 1

# Sort the data by time
sorted_times = sorted(req_counts.keys())
# X-axis: Seconds from the start of the log
X = np.array(sorted_times) - sorted_times[0] 
# Y-axis: Number of requests per second
y = np.array([req_counts[t] for t in sorted_times]) 

# 2. Regression Analysis (Linear Regression / Polynomial degree 1)
# Calculate the trend line: y = mx + c
m, c = np.polyfit(X, y, 1)
y_pred = m * X + c

# 3. Anomaly (DDoS) Detection using Residuals
residuals = y - y_pred
std_dev = np.std(residuals)
# Define threshold for anomaly (Mean + 3 Standard Deviations)
threshold = np.mean(residuals) + 3 * std_dev 

anomalies_x = []
anomalies_y = []
attack_times = []

for i in range(len(X)):
    if residuals[i] > threshold:
        anomalies_x.append(X[i])
        anomalies_y.append(y[i])
        attack_times.append(sorted_times[i])

if attack_times:
    start_attack = datetime.fromtimestamp(min(attack_times)).strftime('%Y-%m-%d %H:%M:%S')
    end_attack = datetime.fromtimestamp(max(attack_times)).strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[!] DDoS attack detected in the following time interval:")
    print(f"Start Time: {start_attack}")
    print(f"End Time: {end_attack}")
else:
    print("\n[OK] No DDoS attack detected.")

# 4. Visualization
plt.figure(figsize=(12, 6))
plt.plot(X, y, label='Requests per Second', color='blue', alpha=0.5)
plt.plot(X, y_pred, label='Regression Trend Line', color='green', linewidth=2)

if anomalies_x:
    plt.scatter(anomalies_x, anomalies_y, color='red', label='Detected DDoS Anomaly', zorder=5)

plt.title('Web Server Log Analysis: DDoS Detection via Regression')
plt.xlabel('Time (Seconds from start)')
plt.ylabel('Requests / Second')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ddos_analysis.png')
print("Visualization saved as: ddos_analysis.png")
