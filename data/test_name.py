import torch

# Load the embeddings from the .pth file
#embeddings = torch.load("20180402-114759-vggface2.pth")

# Get the size of the embeddings tensor
#embeddings_size = embeddings.shape

#print("Embeddings size:", embeddings_size)

import torch

# Đường dẫn tới file .pt
file_path = '20180402-114759-vggface2.pth'

# Load mô hình từ file .pt
model = torch.load(file_path, map_location=torch.device('cpu'))

# In ra shape của từng layer trong mô hình
for name, param in model.items():
    print(f"Layer: {name}, Shape: {param.shape}")

# Nếu mô hình có thuộc tính state_dict, bạn có thể truy cập vào đó để xem chi tiết hơn
if hasattr(model, 'state_dict'):
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        print(f"State_dict key: {key}, Shape: {value.shape}")


import numpy as np

USERNAMES_PATH = 'usernames.npy'

usernames = np.load(USERNAMES_PATH)

print("name size:", usernames.size)
