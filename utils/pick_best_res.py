import json
import math
from sys import argv

import numpy as np

if __name__ == "__main__":
    dir = argv[1]
    metric = argv[2]
    hyperparameters = argv[3:]

    with open(dir, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]
    metric_val = np.zeros(len(data))
    best_config = None

    for i, d in enumerate(data):
        metric_val[i] = d[metric]

    best_exp = np.argmax(metric_val)
    best_config = data[best_exp]

    print(f"### results sorted {np.sort(metric_val)}")
    print("@@@@ Best configuration setup @@@@", flush=True)
    for h in hyperparameters:
        print(f"{h}: {best_config[h]}", flush=True)
    print(f"{metric}: {metric_val[best_exp]}", flush=True)

    num_epochs = len(best_config['train_times'])
    print(f"Num of trained epochs: {num_epochs}")
