# train.py
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import myNN
import time

# ---------- Dataset ----------
class ByteDataset(Dataset):
    def __init__(self, text, context_len):
        self.data = list(text.encode("utf-8"))
        self.context_len = context_len
    
    def __len__(self):
        return len(self.data) - self.context_len  # ✅ prevent out-of-range

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.context_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + self.context_len], dtype=torch.long)

        return x, y

start_time = time.time()
print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

num_epochs = 40
model = myNN()

# ---------- Command-line input ----------
if len(sys.argv) < 2:
    print("Usage: python train.py <training_file>")
    sys.exit(1)

train_file = sys.argv[1]
num_epochs = int(sys.argv[2])
base = os.path.splitext(os.path.basename(train_file))[0]  # e.g. "training-eng"
model_name = f"models/summermute-{base.split('-')[-1]}.pt"  # "summermute-eng.pt"

# ---------- Load data ----------
with open(train_file, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()
print(f"✅ Data read from {train_file}")

dataset = ByteDataset(text, model.context_len)
print("Dataset initialized.")
loader = DataLoader(dataset, batch_size=600, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# ---------- Training ----------
for epoch in range(num_epochs):  # small number for testing
    print(f"Starting epoch {epoch+1}")
    for i, (x, y) in enumerate(loader):
        #print(x, y)
        optimizer.zero_grad()
        out = model(x)
        #print("out:", out)
        loss = loss_fn(out, y)
        # print(epoch, loss)
        loss.backward()
        optimizer.step()
        # Print every 10 batches
        if i % 20 == 0:
            print(f"Epoch {epoch+1}, batch {i}, loss={loss.item():.4f}")

    print(f"Epoch {epoch+1} final loss={loss.item():.4f}")

# ---------- Save model ----------
print(model)
for name, param in model.named_parameters():
    print(name, param.shape)
    print(param)  # prints the actual tensor values
torch.save(model.state_dict(), model_name)
print("Model saved to ", model_name)

end_time = time.time()
duration = end_time - start_time

hours, rem = divmod(duration, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
print(f"Total duration: {int(hours)}h {int(minutes)}m {int(seconds)}s for {num_epochs} epochs")