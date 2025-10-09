import torch
import torch.nn as nn

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

checkpoint = torch.load("models/summermute.pt")
chars = checkpoint["chars"]
idx2char = {i:c for i,c in enumerate(chars)}
char2idx = {c:i for i,c in enumerate(chars)}

model = CharRNN(len(chars))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def get_response(prompt):
    seq = [char2idx[c] for c in prompt]
    hidden = None
    for _ in range(200):
        x = torch.tensor([seq[-25:]])
        out, hidden = model(x, hidden)
        next_idx = torch.argmax(out[0, -1]).item()
        seq.append(next_idx)
    return("".join(idx2char[i] for i in seq))

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["quit", "exit"]:
        break
    prompt = "User: " + user_input + "\nAI: "
    response = get_response(prompt)[len(prompt):].strip()
    print("Summermute:", response)



