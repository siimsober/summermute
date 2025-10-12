# train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import myNN

# ---------- Dataset ----------
class ByteDataset(Dataset):
    def __init__(self, text):
        self.data = list(text.encode("utf-8"))
    
    def __len__(self):
        return len(self.data) - 1  # ✅ prevent out-of-range

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1], dtype=torch.long)
        return x, y

# ---------- Load data ----------
with open("data/training.txt", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()
print("Data read.")

#text = "ab" * 500000
#print("Data generated.")


dataset = ByteDataset(text)
print("Dataset initialized.")
loader = DataLoader(dataset, batch_size=1000, shuffle=True)

model = myNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# ---------- Training ----------
for epoch in range(1):  # small number for testing
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
torch.save(model.state_dict(), "models/summermute.pt")
print("Model saved to models/summermute.pt")