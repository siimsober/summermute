# model.py
import torch
import torch.nn as nn

class myNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.context_len = 32
        self.vocab_size = vocab_size
        embed_size = 7
        hidden_size = 64
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size * self.context_len, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        #print("forward embed:", x)
        x = x.view(x.size(0), -1)  # flatten to [batch, 4*hidden_size]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        #print("forward fc:", x)
        return x
        