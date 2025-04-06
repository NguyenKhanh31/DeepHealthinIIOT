import numpy as np
import pandas as pd
import gc

def data_read(s, r):
    Label = pd.read_csv(r'Datasets\Label.csv')

    vibr_list = ['Normal', 'Fault-1', 'Fault-2', 'Fault-3', 'Fault-4', 'Fault-5']
    vibr_all = {vibr: [] for vibr in vibr_list}

    for vibr in vibr_list:
        vibr_data = pd.read_csv(f'./Datasets/{vibr}.csv', nrows=r).drop('Index', axis=1).reset_index(drop=True).values.reshape([-1, 4000])
        vibr_all[vibr] = [vibr_data[j] for j in range(r)]
        del vibr_data
        gc.collect()

    vibr_all = {k: pd.DataFrame(v).reset_index(drop=True).values.reshape((-1, 1)) for k, v in vibr_all.items()}

    X_dataset = pd.concat([pd.DataFrame(vibr_all[vibr]) for vibr in vibr_list], axis=0).reset_index(drop=True).values.reshape([-1, 1])
    num_samples = (len(vibr_all['Normal']) // s) * 6
    X_dataset = X_dataset[:num_samples * s]
    y_dataset = X_dataset[s:num_samples * s]

    print(num_samples, len(X_dataset), len(y_dataset))

    X_data = X_dataset.reshape((-1, s, 1))
    y_data = y_dataset.reshape((-1, s, 1))

    label = pd.concat([Label[f"L{i}"].iloc[:num_samples // 6] for i in range(6)], axis=0).reset_index(drop=True).values.reshape((-1, 1))

    return X_data, y_data, label

def data_embedding(x, y, seq_len):
    X_data = [pd.DataFrame(x[i]).values.reshape((8, seq_len // 8)) for i in range(len(x))]
    y_data = [pd.DataFrame(y[i]).values.reshape((8, seq_len // 8)) for i in range(len(y))]

    return X_data, y_data
