# train.py
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import myNN

# ---------- Dataset ----------
class ByteDataset(Dataset):
    def __init__(self, text, context_len):
        self.data = list(text.encode("utf-8"))
        self.context_len = context_len
    
    def __len__(self):
        return len(self.data) - self.context_len  # ✅ prevent out-of-range

    def __getitem__(self, idx):
        # input: two consecutive bytes
        x = torch.tensor(self.data[idx:idx + self.context_len], dtype=torch.long)

        # target: the next byte after those two
        y = torch.tensor(self.data[idx + self.context_len], dtype=torch.long)

        return x, y

model = myNN()

# ---------- Command-line input ----------
if len(sys.argv) < 2:
    print("Usage: python train.py <training_file>")
    sys.exit(1)

train_file = sys.argv[1]
base = os.path.splitext(os.path.basename(train_file))[0]  # e.g. "training-eng"
model_name = f"models/summermute-{base.split('-')[-1]}.pt"  # "summermute-eng.pt"

# ---------- Load data ----------
with open(train_file, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()
print(f"✅ Data read from {train_file}")

dataset = ByteDataset(text, model.context_len)
print("Dataset initialized.")
loader = DataLoader(dataset, batch_size=500, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# ---------- Training ----------
for epoch in range(20):  # small number for testing
    print(f"Starting epoch {epoch+1}")
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        print(loss)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss={loss.item()}")

# ---------- Save model ----------
print(model)
for name, param in model.named_parameters():
    print(name, param.shape)
    print(param)  # prints the actual tensor values
torch.save(model.state_dict(), model_name)
print("Model saved to ", model_name)