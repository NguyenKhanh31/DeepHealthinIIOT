


import pandas as pd 
import numpy as np 
import os
import data_embedding, data_read, modules_test
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Conv1D, Flatten
from sklearn.preprocessing import OneHotEncoder
import time




seq_length = 1024
line_num = 1000

# Data
X_data, y_data = data_read.data_read(seq_length, line_num)
X_data = data_embedding.data_embedding(X_data, seq_length)
print("Total data volume: {}".format(len(X_data)))

# Shuffle
Data = list(zip(X_data, y_data))
random.shuffle(Data)
X_data, y_data = zip(*Data)
X_data, y_data = np.array(X_data), np.array(y_data)

# Data split
X_train, y_train = X_data[0:int(len(X_data)*0.7)-1], y_data[0:int(len(y_data)*0.7)-1]
X_valuate, y_valuate = X_data[int(len(X_data)*0.7):int(len(X_data)*0.9)-1], y_data[int(len(X_data)*0.7):int(len(X_data)*0.9)-1]
X_test, y_test = X_data[int(len(X_data)*0.9):len(X_data)-1], y_data[int(len(X_data)*0.9):len(y_data)-1]
print("Train data volume: {}".format(len(X_train)), "Valuate data volume: {}".format(len(X_valuate)), "Teat data volume: {}".format(len(X_test)))


# Hyperparameters
batch_size = 128
lr = 0.0001
hidden_units = seq_length // 8
maxlen = 8
num_blocks = 3
num_epochs = 300
num_heads = 8
dropout_rate = 0.1
lambda_loss_amount = 0.0015

class NormalizeLayer(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8):
        super(NormalizeLayer, self).__init__()
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=epsilon)

    def call(self, inputs):
        return self.layer_norm(inputs)

class MultiheadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_units, num_heads=num_heads, dropout_rate=dropout_rate):
        super(MultiheadAttentionLayer, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.dense_Q = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.dense_K = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.dense_V = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.normalize = NormalizeLayer()

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

        scores = tf.matmul(Q_, K_, transpose_b=True) / tf.math.sqrt(tf.cast(K_.shape[-1], tf.float32))
        attention_weights = tf.nn.softmax(scores)

        query_masks = tf.cast(tf.reduce_sum(tf.abs(queries), axis=-1, keepdims=True) > 0, tf.float32)
        query_masks = tf.tile(query_masks, [self.num_heads, 1, tf.shape(keys)[1]])
        attention_weights *= query_masks

        outputs = tf.matmul(attention_weights, V_)
        outputs = self.merge_heads(outputs)
        outputs += queries

        return self.normalize(outputs)

class FeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, num_units):
        super(FeedForwardLayer, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=num_units[0], kernel_size=1, activation=tf.nn.relu, use_bias=True)
        self.conv2 = tf.keras.layers.Conv1D(filters=num_units[1], kernel_size=1, activation=None, use_bias=True)
        self.normalize = NormalizeLayer()

    def call(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs += inputs
        return self.normalize(outputs)

def one_hot_encoding(y_):
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_ = y_.reshape(-1, 1)
    return encoder.fit_transform(y_)

class MultiheadAttentionLayerWrapper(tf.keras.layers.Layer):
    def __init__(self, num_units, num_heads=8, dropout_rate=0.1, **kwargs):
        # Các tham số bổ sung như 'name', 'trainable', 'dtype' sẽ ở lại trong kwargs
        super(MultiheadAttentionLayerWrapper, self).__init__(**kwargs)
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.multihead_attention = MultiheadAttentionLayer(num_units, num_heads, dropout_rate)

    def get_config(self):
        config = super(MultiheadAttentionLayerWrapper, self).get_config()
        config.update({
            'num_units': self.num_units,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config

    def call(self, inputs):
        queries, keys = inputs
        return self.multihead_attention(queries, keys)


# Define model
def build_model():
    inputs = tf.keras.Input(shape=(maxlen, hidden_units))
    enc = inputs
    for _ in range(num_blocks):
        enc = MultiheadAttentionLayerWrapper(hidden_units)([enc, enc])
    outputs = tf.keras.layers.Dense(6, activation='softmax')(tf.keras.layers.Flatten()(enc))
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = build_model()
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Use existing data
y_train = one_hot_encoding(y_train)
y_val = one_hot_encoding(y_valuate)
y_test = one_hot_encoding(y_test)

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
model.save("DH1_model.keras")
