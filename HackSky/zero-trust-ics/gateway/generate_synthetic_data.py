import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))

def generate_data(num_normal=950, num_anomalies=50):
    data = []
    current_time = datetime.now()

    # Normal data
    for _ in range(num_normal):
        entry = {
            "timestamp": current_time.isoformat(),
            "ip_address": generate_ip(),
            "port": random.choice([502, 44818, 20000]),
            "packet_size": np.random.normal(loc=500, scale=50),
            "response_time": np.random.normal(loc=20, scale=5),
            "command_type": random.choice(["READ", "WRITE"]),
            "value_range": np.random.uniform(0, 100)
        }
        data.append(entry)
        current_time += timedelta(seconds=random.randint(1, 5))

    # Anomalous data
    for _ in range(num_anomalies):
        entry = {
            "timestamp": current_time.isoformat(),
            "ip_address": generate_ip(),
            "port": random.choice([21, 23, 1337]),  # suspicious ports
            "packet_size": np.random.uniform(1000, 2000),  # large size
            "response_time": np.random.uniform(100, 500),  # slow response
            "command_type": random.choice(["ERASE", "OVERRIDE"]),  # rare
            "value_range": np.random.uniform(-100, 300)  # out of range
        }
        data.append(entry)
        current_time += timedelta(seconds=random.randint(1, 5))

    df = pd.DataFrame(data)
    df.to_csv("synthetic_ics_data.csv", index=False)
    print("✅ Synthetic data saved as 'synthetic_ics_data.csv'")

if __name__ == "__main__":
    generate_data()
