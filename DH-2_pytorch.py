import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import data_read_2
from torch.utils.data import TensorDataset, DataLoader

# Chọn thiết bị tính toán (GPU nếu có, ngược lại CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thiết lập tham số
seq_length = 1024
line_num = 1000

# ---------------------- #
#       Dữ liệu         #
# ---------------------- #               

# Đọc dữ liệu (giả sử data_read_2 có các hàm data_read và data_embedding)
X_data, y_data, Label = data_read_2.data_read(seq_length, line_num)
X_data, y_data = data_read_2.data_embedding(X_data, y_data, seq_length)
print("Total data volume: {}".format(len(X_data)))

# Trộn dữ liệu
Data = list(zip(X_data, y_data, Label))  

# Chia dữ liệu thành Train, Validation, Test
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
#   Tham số Hyper-params #
# ---------------------- #

batch_size = 128
lr = 0.0001
hidden_units = seq_length // 8  # 1024//8 = 128
maxlen = 8
num_encoder_layers = 3
num_decoder_layers = 3
num_epochs = 300
num_heads = 8
dropout_rate = 0.1
lambda_loss_amount = 0.0015

# ---------------------- #
#   Định nghĩa module   #
# ---------------------- #

# Module MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate, causality=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"
        self.d_k = d_model // num_heads

        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_K = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)
        self.causality = causality

    def forward(self, queries, keys, values):
        # queries, keys, values có shape: (batch, seq_len, d_model)
        batch_size = queries.size(0)

        Q = self.linear_Q(queries)  # (batch, seq_len, d_model)
        K = self.linear_K(keys)
        V = self.linear_V(values)

        # Tách ra theo số head: (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Tính toán attention scores với scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)  # (batch, num_heads, seq_len, seq_len)

        if self.causality:
            # Tạo mặt nạ causal (chỉ cho phép truy cập thông tin quá khứ)
            seq_len = scores.size(-1)
            mask = torch.tril(torch.ones((seq_len, seq_len), device=scores.device)).bool()
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)  # (batch, num_heads, seq_len, d_k)

        # Nối lại các head
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Kết hợp residual connection và layer normalization
        output = self.layer_norm(output + queries)
        return output

# Module FeedForward
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout_rate):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

# Encoder block
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate, causality=False)
        self.ffn = FeedForward(d_model, dropout_rate)

    def forward(self, x):
        x = self.mha(x, x, x)
        x = self.ffn(x)
        return x

# Decoder block
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super(DecoderBlock, self).__init__()
        # Self-attention với causal mask
        self.mha1 = MultiHeadAttention(d_model, num_heads, dropout_rate, causality=True)
        # Cross-attention với đầu ra của encoder
        self.mha2 = MultiHeadAttention(d_model, num_heads, dropout_rate, causality=False)
        self.ffn = FeedForward(d_model, dropout_rate)

    def forward(self, x, enc_output):
        x = self.mha1(x, x, x)
        x = self.mha2(x, enc_output, enc_output)
        x = self.ffn(x)
        return x

# Lớp Linear cuối cùng
class LinearLayer(nn.Module):
    def __init__(self, d_model):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.linear(x) * self.output_scale

# Mô hình Transformer
class TransformerModel(nn.Module):
    def __init__(self, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_rate, lambda_loss_amount):
        super(TransformerModel, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, dropout_rate) for _ in range(num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, dropout_rate) for _ in range(num_decoder_layers)]
        )
        self.linear_layer = LinearLayer(d_model)
        self.lambda_loss_amount = lambda_loss_amount

    def forward(self, x, y):
        # x: đầu vào encoder, y: đầu vào decoder với shape (batch, seq_len, d_model)
        for enc in self.encoder_layers:
            x = enc(x)
        enc_output = x
        for dec in self.decoder_layers:
            y = dec(y, enc_output)
        pred = self.linear_layer(y)
        return pred

# ---------------------- #
#     Chuẩn bị dữ liệu   #
# ---------------------- #

# Chuyển đổi dữ liệu sang tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Tạo DataLoader cho train, validation, test
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------------------- #
#     Huấn luyện mô hình  #
# ---------------------- #

# Khởi tạo mô hình, hàm loss và optimizer
model = TransformerModel(d_model=hidden_units, num_heads=num_heads,
                         num_encoder_layers=num_encoder_layers,
                         num_decoder_layers=num_decoder_layers,
                         dropout_rate=dropout_rate,
                         lambda_loss_amount=lambda_loss_amount).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Training...\n")
start_time = time.time()
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        # Ở đây dùng cùng dữ liệu cho encoder và decoder (theo cách làm của code gốc)
        output = model(batch_x, batch_y)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)
    epoch_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
end_time = time.time()
print(f"Training time: {end_time - start_time:.3f} seconds")

# Lưu mô hình
model_path = f'./DH2-{seq_length}.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved as {model_path}")

# Đánh giá trên tập validation
model.eval()
val_loss_total = 0.0
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        output = model(batch_x, batch_y)
        loss = criterion(output, batch_y)
        val_loss_total += loss.item() * batch_x.size(0)
val_loss = val_loss_total / len(val_loader.dataset)
print(f"Validation Loss: {val_loss:.4f}")

# Tải mô hình đã lưu (để demo)
model_loaded = TransformerModel(d_model=hidden_units, num_heads=num_heads,
                                num_encoder_layers=num_encoder_layers,
                                num_decoder_layers=num_decoder_layers,
                                dropout_rate=dropout_rate,
                                lambda_loss_amount=lambda_loss_amount).to(device)
model_loaded.load_state_dict(torch.load(model_path))
model_loaded.eval()

# ---------------------- #
#        Kiểm tra       #
# ---------------------- #

test_time_start = time.time()
all_preds = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model_loaded(batch_x, batch_y)
        all_preds.append(pred.cpu())
all_preds = torch.cat(all_preds, dim=0)
test_time_end = time.time()
test_time = test_time_end - test_time_start

# Tính toán RMSE và MAE
mse = criterion(all_preds, torch.tensor(y_test, dtype=torch.float32))
rmse = torch.sqrt(mse).item()
mae = F.l1_loss(all_preds, torch.tensor(y_test, dtype=torch.float32)).item()

print(f"The final RMSE = {rmse:.4f}, The final MAE = {mae:.4f}")
print(f"Test time: {test_time:.3f} seconds")


