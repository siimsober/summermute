# train.py
import sys
import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from s_model import myNN
import time
from tokenizer import BPETokenizer

# ---------- Dataset ----------
class ByteDataset(Dataset):
    def __init__(self, token_ids, context_len):
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_len = context_len
    
    def __len__(self):
        return len(self.tokens) - self.context_len  # ✅ prevent out-of-range

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.context_len] # self.tokens is torch.tensor(dtype=torch.long)
        y = self.tokens[idx+1: idx+self.context_len+1]
        return x, y


# ---------- Vocab Finder ----------
def find_latest_vocab_number(vocab_dir="data/vocab", lang="eng"):
    """
    Find the largest number XXX in vocab-{lang}-XXX.tsv files.
    If none exist, return 0.
    """
    pattern = re.compile(rf"vocab-{re.escape(lang)}-(\d+)\.tsv$")
    max_num = 0
    for fname in os.listdir(vocab_dir):
        match = pattern.match(fname)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num


# ---------- Command-line input ----------
if len(sys.argv) < 3:
    print("Usage: python train.py <training_file> <epochs>")
    sys.exit(1)

train_file = sys.argv[1]
num_epochs = int(sys.argv[2])
base = os.path.splitext(os.path.basename(train_file))[0]  # e.g. "training-eng"
lang = base.split('-')[-1]
model_name = f"models/summermute-{lang}.pt"  # "summermute-eng.pt"

# ---------- Load vocab & model ----------
latest_num = find_latest_vocab_number("data/vocab", lang)
vocab_path = f"data/vocab/vocab-{lang}-{latest_num}.tsv"

if not os.path.exists(vocab_path):
    print(f"❌ Error: No vocab file found at {vocab_path}")
    sys.exit(1)

tokenizer = BPETokenizer(vocab_path)
vocab_size = tokenizer.get_vocab_size()
model = myNN(vocab_size)

# ---------- Start timing ----------
start_time = time.time()
print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

# ---------- Read training data ----------
with open(train_file, "rb") as f:
    data = f.read()
print(f"✅ Read {len(data)} bytes from {train_file}")

block_size = 64
dataset = ByteDataset(tokenizer.encode(data), block_size)
print(f"✅ Dataset initialized with {len(dataset)} samples")
loader = DataLoader(dataset, batch_size=600, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# ---------- Training ----------
for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
    total_loss = 0
    for i, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % 20 == 0:
            print(f"Batch {i}, loss={loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"✅ Epoch {epoch + 1} complete. Avg loss={avg_loss:.4f}")

# ---------- Save model ----------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), model_name)
print(f"\n✅ Model saved to {model_name}")

# ---------- Timing summary ----------
end_time = time.time()
duration = end_time - start_time
hours, rem = divmod(duration, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
print(f"⏱ Total duration: {int(hours)}h {int(minutes)}m {int(seconds)}s for {num_epochs} epochs")