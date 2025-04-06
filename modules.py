import tensorflow as tf
import random
import numpy as np
import time
import data_read, data_embedding
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Conv1D, Flatten
from sklearn.preprocessing import OneHotEncoder

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

# Hyper-parameters
batch_size = 128
lr = 0.0001
hidden_units = seq_length / 8
maxlen = 8
num_blocks = 3
num_epochs = 300
num_heads = 8
dropout_rate = 0.1
lambda_loss_amount = 0.0015

def multihead_attention(queries, keys, num_units=None, num_heads=8, dropout_rate=0.1,
                        is_training=True, causality=False, scope="multihead_attention"):
    with tf.name_scope(scope):
        # Set default num_units if not provided
        if num_units is None:
            num_units = queries.shape[-1]

        # Linear projections
        Q = Dense(num_units, activation=tf.nn.relu)(queries)
        K = Dense(num_units, activation=tf.nn.relu)(keys)
        V = Dense(num_units, activation=tf.nn.relu)(keys)
        
        # Split and concatenate heads
        Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)
        
        # Scaled Dot-Product Attention
        scores = tf.matmul(Q_, K_, transpose_b=True) / tf.math.sqrt(tf.cast(K_.shape[-1], tf.float32))
        
        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1, keepdims=True))  # Shape: (batch_size, seq_len, 1)
        key_masks = tf.tile(key_masks, [num_heads, 1, 1])  # Expand for multiple heads
        paddings = tf.fill(tf.shape(scores), float('-inf'))
        scores = tf.where(tf.equal(key_masks, 0), paddings, scores)
        
        # Causality Masking (Look-ahead Mask)
        if causality:
            seq_len = tf.shape(queries)[1]
            mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # Lower triangular matrix
            mask = tf.tile(tf.expand_dims(mask, 0), [tf.shape(scores)[0], 1, 1])
            scores = tf.where(tf.equal(mask, 0), paddings, scores)
        
        # Apply Softmax
        attention_weights = tf.nn.softmax(scores)
        
        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1, keepdims=True))
        query_masks = tf.tile(query_masks, [num_heads, 1, tf.shape(keys)[1]])
        attention_weights *= query_masks
        
        # Dropout
        attention_weights = Dropout(rate=dropout_rate)(attention_weights, training=is_training)
        
        # Compute weighted sum
        outputs = tf.matmul(attention_weights, V_)
        
        # Restore shape: Concatenate heads
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)
        
        # Residual connection
        outputs += queries
        
        # Normalize
        outputs = LayerNormalization(epsilon=1e-8)(outputs)
    
    return outputs

def feedforward(inputs, num_units, scope="feedforward"):
    with tf.name_scope(scope):
        # Inner layer
        outputs = Conv1D(filters=num_units[0], kernel_size=1, activation=tf.nn.relu, use_bias=True)(inputs)
        
        # Readout layer
        outputs = Conv1D(filters=num_units[1], kernel_size=1, activation=None, use_bias=True)(outputs)
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = LayerNormalization(epsilon=1e-6)(outputs)
    
    return outputs

def attention_block(inputs, hidden_units, num_heads):
    enc = multihead_attention(queries=inputs,
                              keys=inputs,
                              num_units=hidden_units,
                              num_heads=num_heads,
                              dropout_rate=0.1,
                              is_training=True,
                              causality=False)
    enc = feedforward(enc, num_units=[4 * hidden_units, hidden_units])
    
    return enc

def linear(seq_len, inputs):
    logits = Flatten()(inputs)  # Flatten the inputs
    
    fc_W = tf.Variable(tf.random.truncated_normal(shape=(seq_len, 6), mean=0, stddev=0.1))
    fc_b = tf.Variable(tf.zeros(6))
    
    logits = tf.matmul(logits, fc_W) + fc_b
    
    return logits

def one_hot_encoding(y_):
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    return encoder.fit_transform(np.array(y_).reshape(-1, 1))

    class AttentionModel(tf.keras.Model):
        def __init__(self, hidden_units, num_heads, seq_length, lambda_loss_amount):
            super(AttentionModel, self).__init__()
            self.hidden_units = hidden_units
            self.num_heads = num_heads
            self.seq_length = seq_length
            self.lambda_loss_amount = lambda_loss_amount
            
            # Define layers
            self.attention1 = self.attention_block()
            self.attention2 = self.attention_block()
            self.attention3 = self.attention_block()
            self.dense_out = tf.keras.layers.Dense(seq_length, activation=None)
        
        def attention_block(self):
            return tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.hidden_units)
        
        def call(self, inputs, training=False):
            enc1 = self.attention1(inputs, inputs)
            enc2 = self.attention2(enc1, enc1)
            enc3 = self.attention3(enc2, enc2)
            pred = self.dense_out(enc3)
            return pred

    # Model initialization
    model = AttentionModel(hidden_units, num_heads, seq_length, lambda_loss_amount)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Training step
    def train_step(model, x, y):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = loss_object(y, pred)
            l2_loss = lambda_loss_amount * tf.add_n([tf.nn.l2_loss(var) for var in model.trainable_variables])
            total_loss = loss + l2_loss
        
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        return total_loss, accuracy

# Lists to store metrics
test_losses = []
test_accuracies = []
valuate_accuracies = []
valuate_losses = []
train_losses = []
train_accuracies = []
confusion_matrixes = []
train_time, val_time, test_time = [], [], []

def train_and_evaluate(model, X_train, y_train, X_valuate, y_valuate, batch_size, num_epochs):
    num_examples = len(X_train)
    train_accuracies = []
    train_losses = []
    valuate_accuracies = []
    valuate_losses = []
    val_time = []
    
    print("Training...")
    for i in range(num_epochs):
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            history = model.train_on_batch(batch_x, batch_y)
            train_losses.append(history[0])
            train_accuracies.append(history[1])
        
        validation_time_start = time.time()
        valuate_accuracy, valuate_loss = evaluate(X_valuate, y_valuate, model, batch_size)
        validation_time_end = time.time()
        val_time.append(validation_time_end - validation_time_start)
        valuate_accuracies.append(valuate_accuracy)
        valuate_losses.append(valuate_loss)
        
        print(f"EPOCH {i + 1} ...")
        print(f"Valuate Accuracy = {valuate_accuracy:.4f} Valuate Loss = {valuate_loss:.4f} Validation time = {validation_time_end - validation_time_start:.3f}")
        print()
    
    model.save(f'./DH1-{batch_size}.h5')
    print("Model saved")

def test_model(model, X_test, y_test, batch_size):
    test_time_start = time.time()
    loss, final_acc = model.evaluate(X_test, y_test, verbose=0)
    test_time_end = time.time()
    
    print(f"The Final Test Accuracy = {final_acc:.5f}")
    print(f"The time consumption of test = {test_time_end - test_time_start:.3f}")
    
    return final_acc, loss, test_time_end - test_time_start






