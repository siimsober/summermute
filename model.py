# model.py
import torch
import torch.nn as nn

class myNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_len = 8
        vocab_size = 256
        hidden_size = 5
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.fc = nn.Linear(hidden_size * self.context_len, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        #print("forward embed:", x)
        x = x.view(x.size(0), -1)  # flatten to [batch, 4*hidden_size]
        x = self.fc(x)
        #print("forward fc:", x)
        return x
        