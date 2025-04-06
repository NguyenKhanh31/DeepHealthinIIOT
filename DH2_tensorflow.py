import tensorflow as tf
import numpy as np
import time
import data_read_2

# In ra version của TensorFlow (TensorFlow sẽ tự động sử dụng GPU nếu có)
print("TensorFlow version:", tf.__version__)

# ---------------------- #
#       THAM SỐ        #
# ---------------------- #
seq_length = 1024
line_num = 1000

# ---------------------- #
#       DỮ LIỆU       #
# ---------------------- #
# Đọc dữ liệu (giả sử data_read_2 có các hàm data_read và data_embedding)
X_data, y_data, Label = data_read_2.data_read(seq_length, line_num)
X_data, y_data = data_read_2.data_embedding(X_data, y_data, seq_length)
print("Total data volume: {}".format(len(X_data)))

# Trộn dữ liệu và chia thành Train, Validation, Test
num_total = len(X_data)
train_end = int(num_total * 0.9) - 1
val_end = int(num_total * 0.95) - 1

X_train = X_data[:train_end]
y_train = y_data[:train_end]
X_val = X_data[train_end:val_end]
y_val = y_data[train_end:val_end]
X_test = X_data[val_end:num_total-1]
y_test = y_data[val_end:num_total-1]
Label_test = Label[val_end:num_total-1]

print("Train data volume: {}, Val data volume: {}, Test data volume: {}"
      .format(len(X_train), len(X_val), len(X_test)))

# ---------------------- #
#  THAM SỐ HYPERPARAMS  #
# ---------------------- #
batch_size = 128
lr = 0.0001
hidden_units = seq_length // 8  # 1024//8 = 128
maxlen = 8
num_encoder_layers = 3
num_decoder_layers = 3
num_epochs = 2
num_heads = 8
dropout_rate = 0.1
lambda_loss_amount = 0.0015

# ---------------------- #
#  CHUẨN BỊ DỮ LIỆU    #
# ---------------------- #
# Chuyển đổi dữ liệu sang numpy array và ép kiểu về float32
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_val   = np.array(X_val, dtype=np.float32)
y_val   = np.array(y_val, dtype=np.float32)
X_test  = np.array(X_test, dtype=np.float32)
y_test  = np.array(y_test, dtype=np.float32)

# Tạo tf.data.Dataset cho train, validation và test
train_inputs = (X_train, y_train)
train_targets = y_train
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size)

val_inputs = (X_val, y_val)
val_targets = y_val
val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_targets)).batch(batch_size)

test_inputs = (X_test, y_test)
test_targets = y_test
test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_targets)).batch(32)

# ---------------------- #
#    ĐỊNH NGHĨA MODEL   #
# ---------------------- #

# MultiHeadAttention custom layer
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate, causality=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        
        self.linear_Q = tf.keras.layers.Dense(d_model)
        self.linear_K = tf.keras.layers.Dense(d_model)
        self.linear_V = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.causality = causality

    def call(self, queries, keys, values, training):
        # queries, keys, values: shape (batch, seq_len, d_model)
        batch_size = tf.shape(queries)[0]
        Q = self.linear_Q(queries)  # (batch, seq_len, d_model)
        K = self.linear_K(keys)
        V = self.linear_V(values)
        
        # Tách ra theo số head: chuyển về shape (batch, num_heads, seq_len, d_k)
        def split_heads(x):
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_k))
            return tf.transpose(x, perm=[0, 2, 1, 3])
        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)
        
        # Tính attention scores với scaled dot-product
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        if self.causality:
            # Tạo mask causal: chỉ cho phép truy cập thông tin quá khứ
            seq_len = tf.shape(scores)[-1]
            mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            mask = tf.cast(mask, tf.bool)
            scores = tf.where(mask, scores, tf.fill(tf.shape(scores), -1e9))
        attn = tf.nn.softmax(scores, axis=-1)
        attn = self.dropout(attn, training=training)
        output = tf.matmul(attn, V)  # (batch, num_heads, seq_len, d_k)
        
        # Nối lại các head
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch, seq_len, num_heads, d_k)
        output = tf.reshape(output, (batch_size, -1, self.d_model))  # (batch, seq_len, d_model)
        
        # Kết hợp residual connection và layer normalization
        output = self.layer_norm(output + queries)
        return output

# FeedForward layer
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout_rate):
        super(FeedForward, self).__init__()
        self.linear1 = tf.keras.layers.Dense(4 * d_model)
        self.linear2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, x, training):
        residual = x
        x = self.linear1(x)
        x = tf.nn.relu(x)
        x = self.linear2(x)
        x = self.dropout(x, training=training)
        x = self.layer_norm(x + residual)
        return x

# Encoder block
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate, causality=False)
        self.ffn = FeedForward(d_model, dropout_rate)
        
    def call(self, x, training):
        x = self.mha(x, x, x, training=training)
        x = self.ffn(x, training=training)
        return x

# Decoder block
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate):
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads, dropout_rate, causality=True)
        self.mha2 = MultiHeadAttention(d_model, num_heads, dropout_rate, causality=False)
        self.ffn = FeedForward(d_model, dropout_rate)
        
    def call(self, x, enc_output, training):
        x = self.mha1(x, x, x, training=training)
        x = self.mha2(x, enc_output, enc_output, training=training)
        x = self.ffn(x, training=training)
        return x

# Linear layer cuối cùng
class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(LinearLayer, self).__init__()
        self.linear = tf.keras.layers.Dense(d_model)
        self.output_scale = self.add_weight(name="output_scale", shape=(), initializer=tf.keras.initializers.Ones())
        
    def call(self, x):
        return self.linear(x) * self.output_scale

# Mô hình Transformer chính
class TransformerModel(tf.keras.Model):
    def __init__(self, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_rate, lambda_loss_amount):
        super(TransformerModel, self).__init__()
        self.encoder_layers = [EncoderBlock(d_model, num_heads, dropout_rate) for _ in range(num_encoder_layers)]
        self.decoder_layers = [DecoderBlock(d_model, num_heads, dropout_rate) for _ in range(num_decoder_layers)]
        self.linear_layer = LinearLayer(d_model)
        self.lambda_loss_amount = lambda_loss_amount
        
    def call(self, inputs, training=False):
        # inputs là một tuple chứa (encoder_input, decoder_input)
        encoder_input, decoder_input = inputs
        for enc in self.encoder_layers:
            encoder_input = enc(encoder_input, training=training)
        enc_output = encoder_input
        for dec in self.decoder_layers:
            decoder_input = dec(decoder_input, enc_output, training=training)
        pred = self.linear_layer(decoder_input)
        return pred


# ---------------------- #
#    KHỞI TẠO MODEL    #
# ---------------------- #
model = TransformerModel(d_model=hidden_units, num_heads=num_heads,
                         num_encoder_layers=num_encoder_layers,
                         num_decoder_layers=num_decoder_layers,
                         dropout_rate=dropout_rate,
                         lambda_loss_amount=lambda_loss_amount)

# Biên dịch model với optimizer và loss (MSE)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.MeanSquaredError())

# ---------------------- #
#       HUẤN LUYỆN      #
# ---------------------- #
print("Training...\n")
start_time = time.time()
model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset, verbose=2)
end_time = time.time()
print(f"Training time: {end_time - start_time:.3f} seconds")

# Lưu model (chỉ lưu weights)
model_path = f'./DH2-{seq_length}.weights.h5'
model.save_weights(model_path)
print(f"Model saved as {model_path}")

# Đánh giá trên tập validation
val_loss = model.evaluate(val_dataset, verbose=0)
print(f"Validation Loss: {val_loss:.4f}")

# Tải lại model đã lưu để demo
model_loaded = TransformerModel(d_model=hidden_units, num_heads=num_heads,
                                num_encoder_layers=num_encoder_layers,
                                num_decoder_layers=num_decoder_layers,
                                dropout_rate=dropout_rate,
                                lambda_loss_amount=lambda_loss_amount)
# Đảm bảo xây dựng model bằng cách chạy một batch mẫu:
dummy_encoder = tf.random.uniform((1, X_train.shape[1], hidden_units))
dummy_decoder = tf.random.uniform((1, y_train.shape[1], hidden_units))
_ = model_loaded((dummy_encoder, dummy_decoder))
model_loaded.load_weights(model_path)

# ---------------------- #
#         KIỂM TRA      #
# ---------------------- #
test_time_start = time.time()
all_preds = []
for (enc_input, dec_input), target in test_dataset:
    preds = model_loaded((enc_input, dec_input), training=False)
    all_preds.append(preds)
all_preds = tf.concat(all_preds, axis=0)
test_time_end = time.time()
test_time = test_time_end - test_time_start


# Tính RMSE và MAE
mse = tf.keras.losses.MeanSquaredError()(y_test, all_preds)
rmse = tf.sqrt(mse).numpy()
mae = tf.keras.losses.MeanAbsoluteError()(y_test, all_preds).numpy()

print(f"The final RMSE = {rmse:.4f}, The final MAE = {mae:.4f}")
print(f"Test time: {test_time:.3f} seconds")
