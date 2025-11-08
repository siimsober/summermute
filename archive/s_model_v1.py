# model.py
import torch
import torch.nn as nn

class myNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        embed_size = 16
        hidden_size = 64
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x) # [batch, seq_len, embed_size]
        #print("forward embed:", x)
        x, hidden = self.rnn(x, hidden)
        logits = self.fc(x)
        #print("forward fc:", x)
        return logits, hidden
        