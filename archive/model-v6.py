# model.py
import torch
import torch.nn as nn

class myNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_len = 4
        vocab_size = 256
        self.fc = nn.Linear(vocab_size * self.context_len, vocab_size)

    def forward(self, x):
        # x shape: [batch, 4]
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x1_onehot = nn.functional.one_hot(x1, num_classes=256).float()
        x2_onehot = nn.functional.one_hot(x2, num_classes=256).float()
        x3_onehot = nn.functional.one_hot(x3, num_classes=256).float()
        x4_onehot = nn.functional.one_hot(x4, num_classes=256).float()
        x = torch.cat([x1_onehot, x2_onehot, x3_onehot, x4_onehot], dim=-1)  # [batch, 512]
        return self.fc(x)  # [batch, 256] logits