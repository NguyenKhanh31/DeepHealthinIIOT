import tensorflow as tf
import random
import numpy as np
import time
import data_read_2
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Conv1D, Flatten


seq_length = 1024
line_num = 1000

# Data
X_data, y_data, Label = data_read_2.data_read(seq_length, line_num)
X_data, y_data = data_read_2.data_embedding(X_data, y_data, seq_length)
print("Total data volume: {}".format(len(X_data)))

# Shuffle
Data = list(zip(X_data, y_data, Label))
random.shuffle(Data)
X_data, y_data, Label = zip(*Data)
X_data, y_data, Label = np.array(X_data), np.array(y_data), np.array(Label)

# Data split
X_train, y_train = X_data[0:int(len(X_data)*0.9)-1], y_data[0:int(len(y_data)*0.9)-1]
X_valuate, y_valuate = X_data[int(len(X_data)*0.9):int(len(X_data)*0.95)-1], y_data[int(len(X_data)*0.9):int(len(X_data)*0.95)-1]
X_test, y_test, Label_test = X_data[int(len(X_data)*0.95):len(X_data)-1], y_data[int(len(X_data)*0.95):len(y_data)-1], Label[int(len(X_data)*0.95):len(y_data)-1]
print("Train data volume: {}".format(len(X_train)), "Valuate data volume: {}".format(len(X_valuate)), "Test data volume: {}".format(len(X_test)))

# Hyper-parameters
batch_size = 128
lr = 0.0001
hidden_units = seq_length / 8
maxlen = 8
num_blocks = 3
num_epochs = 2
num_heads = 8
dropout_rate = 0.1
lambda_loss_amount = 0.0015

#module
def ln(inputs, epsilon=1e-8):
    layer_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon)
    return layer_norm(inputs)

def mask(inputs, queries=None, keys=None, type=None):
    padding_num = -1e9  # Sử dụng giá trị nhỏ thay vì -2**32 + 1 để tránh tràn số
    
    if type in ("k", "key", "keys"):
        masks = tf.cast(tf.math.reduce_sum(tf.abs(keys), axis=-1) > 0, tf.float32)
        masks = tf.expand_dims(masks, 1)
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])
        outputs = tf.where(tf.equal(masks, 0), tf.ones_like(inputs) * padding_num, inputs)
    
    elif type in ("q", "query", "queries"):
        masks = tf.cast(tf.math.reduce_sum(tf.abs(queries), axis=-1) > 0, tf.float32)
        masks = tf.expand_dims(masks, -1)
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])
        outputs = inputs * masks
    
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])
        tril = tf.linalg.band_part(diag_vals, -1, 0)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
        outputs = tf.where(tf.equal(masks, 0), tf.ones_like(masks) * padding_num, inputs)
    
    else:
        raise ValueError("Invalid mask type! Use 'key', 'query', or 'future'.")
    
    return outputs

def scaled_dot_product_attention(Q, K, V, causality=False, dropout_rate=0.5, training=True):
    d_k = tf.cast(tf.shape(K)[-1], tf.float32)

    # Dot product
    outputs = tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1]))

    # Scale
    outputs /= tf.math.sqrt(d_k)

    # Key masking
    outputs = mask(outputs, queries=Q, keys=K, type="key")

    # Causality masking (future masking)
    if causality:
        outputs = mask(outputs, type="future")

    # Apply softmax
    outputs = tf.nn.softmax(outputs, axis=-1)
    
    # Apply dropout if training
    if training and dropout_rate > 0:
        outputs = tf.nn.dropout(outputs, rate=dropout_rate)

    # Query masking
    outputs = mask(outputs, queries=Q, keys=K, type="query")

    # Weighted sum (context vectors)
    outputs = tf.matmul(outputs, V)
    
    return outputs

def multihead_attention(queries, keys, values,
                        num_heads=8,
                        dropout_rate=0.5,
                        training=True,
                        causality=False):
    d_model = int(tf.shape(queries)[-1])
    
    # Linear projections
    Q = tf.keras.layers.Dense(d_model)(queries)
    K = tf.keras.layers.Dense(d_model)(keys)
    V = tf.keras.layers.Dense(d_model)(values)
    
    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)
    
    # Attention
    outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)
    
    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)
    
    # Residual connection
    outputs += queries
    
    # Normalize
    outputs = ln(outputs)
    
    return outputs

def feedforward(inputs, num_units):
    # Inner layer (Dense with ReLU activation)
    outputs = tf.keras.layers.Conv1D(filters=num_units[0], kernel_size=1, activation=tf.nn.relu, use_bias=True)(inputs)
    
    # Readout layer
    outputs = tf.keras.layers.Conv1D(filters=num_units[1], kernel_size=1, activation=None, use_bias=True)(outputs)
    
    # Residual connection
    outputs += inputs
    
    # Normalize
    outputs = ln(outputs)
    
    return outputs

def positional_encoding(inputs, maxlen, masking=True):
    E = tf.shape(inputs)[-1]
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]
    
    # Position indices
    position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])
    
    # Compute position encoding
    position_enc = np.array([
        [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
        for pos in range(maxlen)
    ])
    
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
    position_enc = tf.convert_to_tensor(position_enc, dtype=tf.float32)
    
    # Lookup encoding
    outputs = tf.nn.embedding_lookup(position_enc, position_ind)
    
    # Apply masking
    if masking:
        outputs = tf.where(tf.equal(inputs, 0), tf.zeros_like(inputs), outputs)
    
    return tf.cast(outputs, dtype=tf.float32)

def encoder_attention_block(inputs, keep_prop):
    enc = multihead_attention(queries=inputs,
                              keys=inputs,
                              values=inputs,
                              dropout_rate=keep_prop)
    enc = feedforward(enc, num_units=[4 * hidden_units, hidden_units])

    return enc

def decoder_attention_block(input1, input2, keep_prop):
    memory = multihead_attention(queries=input1,
                              keys=input1,
                              values=input1,
                              causality=True,
                              dropout_rate=keep_prop)

    dec = multihead_attention(queries=memory,
                              keys=input2,
                              values=input2,
                              dropout_rate=keep_prop)

    dec = feedforward(dec, num_units=[4 * hidden_units, hidden_units])

    return dec

def linear(inputs, seq_length):
    fc_W = tf.Variable(tf.random.truncated_normal(
        shape=(seq_length // 8, seq_length // 8), mean=0, stddev=0.1
    ))
    
    logits = tf.einsum('ntd,dk->ntk', inputs, fc_W)
    output_scale_factor = tf.Variable(1.0, dtype=tf.float32, name="Output_ScaleFactor")

    reshaped_output = output_scale_factor * logits

    return reshaped_output

@tf.function
def evaluate(model, X_data):
    # Dự đoán và tính toán loss + MAE
    loss, mae = model(X_data, training=False)
    
    # Tính tổng loss và MAE
    total_loss = tf.reduce_mean(tf.reduce_sum(loss))
    total_mae = tf.reduce_mean(tf.reduce_sum(mae))

    return total_loss.numpy(), total_mae.numpy()

class TransformerModel(tf.keras.Model):
    def __init__(self, lambda_loss_amount):
        super(TransformerModel, self).__init__()

        self.encoder_block1 = encoder_attention_block
        self.encoder_block2 = encoder_attention_block
        self.encoder_block3 = encoder_attention_block

        self.decoder_block1 = decoder_attention_block
        self.decoder_block2 = decoder_attention_block
        self.decoder_block3 = decoder_attention_block

        self.linear_layer = linear
        self.lambda_loss_amount = lambda_loss_amount

    def call(self, inputs):
        x, y, keep_prob = inputs

        # Encoder
        enc1 = self.encoder_block1(x, keep_prob)
        enc2 = self.encoder_block2(enc1, keep_prob)
        enc3 = self.encoder_block3(enc2, keep_prob)

        # Decoder
        dec1 = self.decoder_block1(y, enc3, keep_prob)
        dec2 = self.decoder_block2(dec1, enc3, keep_prob)
        dec3 = self.decoder_block3(dec2, enc3, keep_prob)

        # Output
        pred = self.linear_layer(dec3)

        return pred

model = TransformerModel(lambda_loss_amount)

# Input tensor
x_input = tf.keras.Input(shape=(maxlen, hidden_units))
y_input = tf.keras.Input(shape=(maxlen, hidden_units))
keep_prob_input = tf.keras.Input(shape=(), dtype=tf.float32)

# Model forward pass
pred_output = model((x_input, y_input, keep_prob_input))

# Loss function
l2_loss = lambda_loss_amount * tf.add_n([tf.nn.l2_loss(var) for var in model.trainable_variables])
loss = tf.sqrt(tf.reduce_mean(tf.nn.l2_loss(y_input - pred_output))) + l2_loss
mae = tf.reduce_mean(tf.abs(y_input - pred_output)) + l2_loss

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=loss)

# Checkpoints
checkpoint = tf.train.Checkpoint(model=model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory="./checkpoints", max_to_keep=3)

test_losses, test_MAEs = [], []
valuate_losses, valuate_MAEs = [], []
train_losses, train_MAEs = [], []
train_time, val_time, test_time = [], [], []

train_losses, train_MAEs = [], []
valuate_losses, valuate_MAEs = [], []
train_time, val_time = [], []

model = TransformerModel(maxlen=seq_length, hidden_units=128, lambda_loss_amount=0.01)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()

# Compile mô hình
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'])

# Bắt đầu huấn luyện
time_start = time.time()

print("Training...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_valuate, y_valuate),
    epochs=num_epochs,
    batch_size=batch_size,
    verbose=1
)

# Lưu thời gian huấn luyện
time_end = time.time()
train_time.append(time_end - time_start)

# Lưu mô hình sau khi huấn luyện
model.save(f'./DH2-{seq_length}')

print(f"Model saved as ./DH2-{seq_length}")
print(f"The time consumption of training stage = {time_end - time_start:.3f} seconds")

# Lưu trữ kết quả đánh giá
valuate_loss, valuate_mae = model.evaluate(X_valuate, y_valuate, batch_size=batch_size)
valuate_losses.append(valuate_loss)
valuate_MAEs.append(valuate_mae)

# Load mô hình đã lưu
model = tf.keras.models.load_model(f'./DH2-{seq_length}')

# Bắt đầu kiểm tra mô hình
test_time_start = time.time()

pred = model.predict(X_test, batch_size=32)

loss = tf.sqrt(tf.reduce_mean(tf.nn.l2_loss(y_test - pred))).numpy()
mae = tf.reduce_mean(tf.abs(y_test - pred)).numpy()

test_time_end = time.time()
test_time.append(test_time_end - test_time_start)
test_losses.append(loss)
test_MAEs.append(mae)

print(f"The final loss = {loss:.4f}, The final MAE = {mae:.4f}")
print(f"The time consumption of test = {test_time_end - test_time_start:.3f} seconds")







