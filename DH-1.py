import tensorflow as tf
import random
import numpy as np
import time
import data_read_2
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Conv1D, Flatten
from sklearn.preprocessing import OneHotEncoder

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

# Module
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

class MultiheadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_units, num_heads=num_heads, dropout_rate=dropout_rate):
        super(MultiheadAttentionLayer, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.dense_Q = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.dense_K = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.dense_V = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.normalize = tf.keras.layers.LayerNormalization()
        self.sdpa = scaled_dot_product_attention

    def split_heads(self, x):
        return tf.concat(tf.split(x, self.num_heads, axis=-1), axis=0)

    def merge_heads(self, x):
        return tf.concat(tf.split(x, self.num_heads, axis=0), axis=-1)

    def call(self, queries, keys):
        Q = self.dense_Q(queries)
        K = self.dense_K(keys)
        V = self.dense_V(keys)

        Q_ = self.split_heads(Q)
        K_ = self.split_heads(K)
        V_ = self.split_heads(V)

        outputs = self.sdpa(Q_, K_, V_, causality=False, dropout_rate=self.dropout_rate, training=True)
        outputs = self.merge_heads(outputs)
        if outputs.shape[-1] != queries.shape[-1]:
            outputs = tf.keras.layers.Dense(queries.shape[-1])(outputs)

        outputs = outputs + queries

        return self.normalize(outputs)

class FeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, num_units, dropout_rate=dropout_rate):
        super(FeedForwardLayer, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(num_units[0], kernel_size=1, activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv1D(num_units[1], kernel_size=1)
        self.normalize = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs += inputs

        return self.normalize(outputs)

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_units, num_heads=num_heads, dropout_rate=dropout_rate):
        super(EncoderBlock, self).__init__()
        self.mha = MultiheadAttentionLayer(num_units[0], num_heads, dropout_rate)
        self.ffn = FeedForwardLayer(num_units, dropout_rate)

    def call(self, inputs):

        print("EncoderBlock.call() is running...")  # Debug
        print(f"EncoderBlock input: {inputs}")  # Kiểm tra giá trị đầu vào
        print(f"EncoderBlock input shape: {inputs.shape}")         
        outputs = self.mha(inputs, inputs)
        outputs = self.ffn(outputs)

        return outputs

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_units, num_heads=num_heads, dropout_rate=dropout_rate):
        super(DecoderBlock, self).__init__()
        self.mmr = MultiheadAttentionLayer(num_units, num_heads, dropout_rate)
        self.mha = MultiheadAttentionLayer(num_units, num_heads, dropout_rate)
        self.ffn = FeedForwardLayer(num_units, dropout_rate)

    def call(self, inputs, enc_outputs):
        outputs = self.mmr(inputs, inputs)
        outputs = self.mha(outputs, enc_outputs)
        outputs = self.ffn(outputs)

        return outputs

# Model
def build_model():
    inputs = tf.keras.layers.Input(shape=(seq_length, 4))
    enc1 = EncoderBlock(num_units=[int(hidden_units*4), int(hidden_units)])(inputs)
    enc2 = EncoderBlock(num_units=[hidden_units*4, hidden_units])(enc1)
    enc3 = EncoderBlock(num_units=[hidden_units*4, hidden_units])(enc2)

    dec1 = DecoderBlock(num_units=[hidden_units*4, hidden_units])(inputs, enc3)
    dec2 = DecoderBlock(num_units=[hidden_units*4, hidden_units])(dec1, enc3)
    dec3 = DecoderBlock(num_units=[hidden_units*4, hidden_units])(dec2, enc3)

    outputs = tf.keras.layers.Dense(4, activation=tf.nn.softmax)(dec3)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)

model = build_model()
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])

# Use existing data
y_train = tf.keras.layers.LayerNormalization(y_train)
y_val = tf.keras.layers.LayerNormalization(y_valuate)
y_test = tf.keras.layers.LayerNormalization(y_test)

# Define X_val
X_val = X_valuate

# Train the model
time_start = time.time()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=num_epochs)
time_end = time.time()
train_time = time_end - time_start
print(f"Training time: {train_time:.3f}s")

# Evaluate on test data
test_time_start = time.time()
test_loss, test_acc = model.evaluate(X_test, y_test)
test_time_end = time.time()
test_time = test_time_end - test_time_start
print(f"Test Accuracy: {test_acc:.5f}, Test Time: {test_time:.3f}s, Train Time: {train_time:.3f}s")

# Save the model
model.save("transformer_model.h5")
