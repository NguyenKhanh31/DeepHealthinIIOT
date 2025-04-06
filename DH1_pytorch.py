import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import time
import math
import data_read, data_embedding

# -------------------------------
# Cài đặt tham số và đọc dữ liệu
# -------------------------------
seq_length = 1024
line_num = 1000

# Data: Giả sử các hàm data_read và data_embedding trả về numpy arrays
X_data, y_data = data_read.data_read(seq_length, line_num)
X_data = data_embedding.data_embedding(X_data, seq_length)
print("Total data volume: {}".format(len(X_data)))

# Shuffle dữ liệu
Data = list(zip(X_data, y_data))
random.shuffle(Data)
X_data, y_data = zip(*Data)
X_data, y_data = np.array(X_data), np.array(y_data)

# Tách dữ liệu: 70% train, 20% validation, 10% test
X_train, y_train = X_data[0:int(len(X_data)*0.7)-1], y_data[0:int(len(y_data)*0.7)-1]
X_valuate, y_valuate = X_data[int(len(X_data)*0.7):int(len(X_data)*0.9)-1], y_data[int(len(X_data)*0.7):int(len(y_data)*0.9)-1]
X_test, y_test = X_data[int(len(X_data)*0.9):len(X_data)-1], y_data[int(len(X_data)*0.9):len(y_data)-1]
print("Train data volume: {}".format(len(X_train)),
      "Valuate data volume: {}".format(len(X_valuate)),
      "Test data volume: {}".format(len(X_test)))

# -------------------------------
# Định nghĩa các siêu tham số
# -------------------------------
batch_size = 128
lr = 0.0001
hidden_units = seq_length // 8
maxlen = 8
num_blocks = 3
num_epochs = 300
num_heads = 8
dropout_rate = 0.1
lambda_loss_amount = 0.0015

# -------------------------------
# Định nghĩa các lớp mô hình
# -------------------------------
class NormalizeLayer(nn.Module):
    def __init__(self, normalized_shape, epsilon=1e-8):
        super(NormalizeLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=epsilon)
    def forward(self, inputs):
        return self.layer_norm(inputs)

class MultiheadAttentionLayer(nn.Module):
    def __init__(self, num_units, num_heads=num_heads, dropout_rate=dropout_rate):
        super(MultiheadAttentionLayer, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.linear_Q = nn.Linear(num_units, num_units)
        self.linear_K = nn.Linear(num_units, num_units)
        self.linear_V = nn.Linear(num_units, num_units)
        self.relu = nn.ReLU()
        self.normalize = NormalizeLayer(num_units)
        self.dropout = nn.Dropout(dropout_rate)
        
    def split_heads(self, x):
        # x: (batch, seq_len, num_units)
        batch_size, seq_len, dim = x.size()
        new_dim = dim // self.num_heads
        x = x.view(batch_size, seq_len, self.num_heads, new_dim)
        # Chuyển đổi: (batch, num_heads, seq_len, new_dim) -> (batch*num_heads, seq_len, new_dim)
        x = x.transpose(1, 2).contiguous().view(batch_size * self.num_heads, seq_len, new_dim)
        return x
    
    def merge_heads(self, x, batch_size):
        # x: (batch*num_heads, seq_len, new_dim)
        new_dim = x.size(-1)
        seq_len = x.size(1)
        x = x.view(batch_size, self.num_heads, seq_len, new_dim)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * new_dim)
        return x
    
    def forward(self, queries, keys):
        # Tính Q, K, V với activation ReLU
        Q = self.relu(self.linear_Q(queries))
        K = self.relu(self.linear_K(keys))
        V = self.relu(self.linear_V(keys))
        
        # Tách thành các head
        Q_ = self.split_heads(Q)
        K_ = self.split_heads(K)
        V_ = self.split_heads(V)
        
        d_k = K_.size(-1)
        scores = torch.bmm(Q_, K_.transpose(1, 2)) / math.sqrt(d_k)  # (batch*num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Áp dụng mask: nếu tổng giá trị tuyệt đối của các phần tử trong vector query = 0 thì mask = 0
        batch_size, seq_len, _ = queries.size()
        key_len = keys.size(1)
        mask = (queries.abs().sum(dim=-1, keepdim=True) > 0).float()  # (batch, seq_len, 1)
        mask = mask.expand(-1, -1, key_len)  # (batch, seq_len, key_len)
        mask = mask.repeat(self.num_heads, 1, 1)  # (batch*num_heads, seq_len, key_len)
        attention_weights = attention_weights * mask
        attention_weights = self.dropout(attention_weights)
        
        outputs = torch.bmm(attention_weights, V_)
        outputs = self.merge_heads(outputs, batch_size)
        outputs = outputs + queries  # Residual connection
        outputs = self.normalize(outputs)
        return outputs

class FeedForwardLayer(nn.Module):
    def __init__(self, num_units):
        """
        num_units: list hoặc tuple [d_ff, hidden_units]
        """
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(num_units[1], num_units[0])
        self.fc2 = nn.Linear(num_units[0], num_units[1])
        self.relu = nn.ReLU()
        self.normalize = NormalizeLayer(num_units[1])
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = out + x
        out = self.normalize(out)
        return out

def one_hot_encoding(y_):
    # Giả sử y_ là numpy array 1D
    y_tensor = torch.tensor(y_, dtype=torch.long)
    num_classes = y_tensor.max().item() + 1
    return F.one_hot(y_tensor, num_classes=num_classes).float()

class MultiheadAttentionLayerWrapper(nn.Module):
    def __init__(self, num_units, num_heads=num_heads, dropout_rate=dropout_rate):
        super(MultiheadAttentionLayerWrapper, self).__init__()
        self.multihead_attention = MultiheadAttentionLayer(num_units, num_heads, dropout_rate)
        
    def forward(self, inputs):
        queries, keys = inputs
        return self.multihead_attention(queries, keys)

# -------------------------------
# Xây dựng mô hình
# -------------------------------
class TransformerModel(nn.Module):
    def __init__(self, maxlen, hidden_units, num_blocks, num_classes=6):
        super(TransformerModel, self).__init__()
        self.layers = nn.ModuleList([MultiheadAttentionLayerWrapper(hidden_units) for _ in range(num_blocks)])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(maxlen * hidden_units, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x: (batch, maxlen, hidden_units)
        for layer in self.layers:
            x = layer((x, x))
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

model = TransformerModel(maxlen, hidden_units, num_blocks)

# -------------------------------
# Định nghĩa hàm loss và optimizer
# -------------------------------
def categorical_crossentropy_loss(outputs, targets):
    # outputs: phân phối xác suất (đã qua softmax), targets: one-hot vector
    loss = - (targets * torch.log(outputs + 1e-8)).sum(dim=1).mean()
    return loss

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -------------------------------
# Sử dụng dữ liệu đã có
# -------------------------------
y_train = one_hot_encoding(y_train)
y_val = one_hot_encoding(y_valuate)
y_test = one_hot_encoding(y_test)
X_val = X_valuate

# Chuyển đổi dữ liệu sang tensor và tạo DataLoader
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train)
val_dataset = TensorDataset(torch.tensor(X_valuate, dtype=torch.float32), y_val)
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# Huấn luyện mô hình
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

time_start = time.time()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = categorical_crossentropy_loss(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    epoch_loss /= len(train_loader.dataset)
    # Có thể in loss của mỗi epoch nếu cần
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
time_end = time.time()
train_time = time_end - time_start
print(f"Training time: {train_time:.3f}s")

# -------------------------------
# Đánh giá trên dữ liệu test
# -------------------------------
model.eval()
test_loss = 0.0
correct = 0
total = 0
test_time_start = time.time()
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        loss = categorical_crossentropy_loss(outputs, batch_y)
        test_loss += loss.item() * batch_X.size(0)
        preds = outputs.argmax(dim=1)
        targets = batch_y.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_X.size(0)
test_loss /= total
test_acc = correct / total
test_time_end = time.time()
test_time = test_time_end - test_time_start
print(f"Test Accuracy: {test_acc:.5f}, Test Time: {test_time:.3f}s, Train Time: {train_time:.3f}s")

# -------------------------------
# Lưu mô hình
# -------------------------------
torch.save(model.state_dict(), "DH1.pth")
