# s_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class myNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = 64
        self.max_len = 256

        self.embed = nn.Embedding(vocab_size, self.embed_size)
        self.pos_embed = nn.Embedding(self.max_len, self.embed_size)

        # trainable weights for fusing previous tokens
        self.fuse = nn.Parameter(torch.randn(self.max_len, self.max_len))
        self.value = nn.Linear(self.embed_size, self.embed_size)  # learned V

        self.fc = nn.Linear(self.embed_size, vocab_size)

    def forward(self, x):
        B, T = x.shape

        # token + positional embeddings
        tok_emb = self.embed(x)                      # [B, T, E]
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pos_embed(pos)[None, :, :]    # [1, T, E]
        x = tok_emb + pos_emb                          # [B, T, E]

        # causal mask to only fuse preceding tokens
        mask = torch.tril(torch.ones(T, T, device=x.device))
        fuse_weights = self.fuse[:T, :T]            # [T, T]
        fuse_weights = fuse_weights.masked_fill(mask == 0, float('-inf'))
        fuse_weights = F.softmax(fuse_weights, dim=-1)  # [T, T]

        V = self.value(x)

        # fuse embeddings
        context = fuse_weights @ V                    # [B, T, E]

        logits = self.fc(context)                     # [B, T, vocab_size]
        return logits
    