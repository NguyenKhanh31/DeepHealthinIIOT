import pandas as pd
def data_embedding(x, seq_len):
    X_data = []
    for i in range(0, len(x)):
        x_data = x[i]
        x_data = pd.DataFrame(x_data).values.reshape((8, int(seq_len / 8)))
        X_data.append(x_data)

    return X_data