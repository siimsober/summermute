# model.py
import torch
import torch.nn as nn

class myNN(nn.Module):
    def __init__(self):
        vocab_size = 256
        super().__init__()
        self.fc = nn.Linear(vocab_size * 2, vocab_size) # 2 char context

    def forward(self, x):
        # x shape: [batch, 2]
        x1 = x[:, 0]
        x2 = x[:, 1]
        x1_onehot = nn.functional.one_hot(x1, num_classes=256).float()
        x2_onehot = nn.functional.one_hot(x2, num_classes=256).float()
        x = torch.cat([x1_onehot, x2_onehot], dim=-1)  # [batch, 512]
        return self.fc(x)  # [batch, 256] logits