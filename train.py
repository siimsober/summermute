# train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- Dataset ----------
class CharDataset(Dataset):
    def __init__(self, text, seq_len=25):
        self.chars = sorted(list(set(text)))
        self.char2idx = {c:i for i,c in enumerate(self.chars)}
        self.idx2char = {i:c for i,c in enumerate(self.chars)}
        self.data = [self.char2idx[c] for c in text]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len])
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1])
        return x, y

# ---------- Model ----------
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

# ---------- Load data ----------
with open("data/training.txt", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

dataset = CharDataset(text)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = CharRNN(len(dataset.chars))
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fn = nn.CrossEntropyLoss()

# ---------- Training ----------
for epoch in range(10):  # small number for testing
    for x, y in loader:
        optimizer.zero_grad()
        out, _ = model(x)
        loss = loss_fn(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

# ---------- Save model ----------
torch.save({
    "model_state_dict": model.state_dict(),
    "chars": dataset.chars
}, "models/summermute.pt")
print("Model saved to models/summermute.pt")