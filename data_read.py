import pandas as pd
import numpy as np
import gc
import pandas as pd
import numpy as np
import gc

def data_read(s, r):
    Label = pd.read_csv('Datasets/Label.csv')

    vibr_list = ['Normal', 'Fault-1', 'Fault-2', 'Fault-3', 'Fault-4', 'Fault-5']
    vibr_all = {vibr: [] for vibr in vibr_list}

    for vibr in vibr_list:
        vibr_data = pd.read_csv(f'Datasets/{vibr}.csv', nrows=r).drop('Index', axis=1).reset_index(drop=True).values.reshape([-1, 4000])
        vibr_all[vibr] = [vibr_data[j] for j in range(r)]

        del vibr_data
        gc.collect()

    for key in vibr_all:
        vibr_all[key] = pd.DataFrame(vibr_all[key]).reset_index(drop=True).values.reshape((-1, 1))

    X_dataset = pd.concat([pd.DataFrame(vibr_all[vibr]) for vibr in vibr_list], axis=0).reset_index(drop=True).values.reshape([-1, 1])

    num_samples = (len(vibr_all['Normal']) // s) * 6
    X_data = pd.concat([pd.DataFrame(X_dataset[:num_samples * s])], axis=1).reset_index(drop=True).values.reshape((-1, s, 1))

    y_data = pd.concat([Label[f"L{i}"].iloc[:num_samples // 6] for i in range(6)], axis=0).reset_index(drop=True).values.reshape((-1, 1))

    return X_data, y_data

import pandas as pd
def data_embedding(x, seq_len):
    X_data = []
    for i in range(0, len(x)):
        x_data = x[i]
        x_data = pd.DataFrame(x_data).values.reshape((8, int(seq_len / 8)))
        X_data.append(x_data)

    return X_data