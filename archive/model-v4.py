# model.py
import torch
import torch.nn as nn

class myNN(nn.Module):
    def __init__(self):
        vocab_size = 256
        super().__init__()
        self.fc = nn.Linear(vocab_size, vocab_size)

    def forward(self, x):
        x_onehot = nn.functional.one_hot(x, num_classes=256).float()
        return self.fc(x_onehot)