# s_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class myNN(nn.Module):
    def __init__(self, vocab_size, embed_size=64, max_len=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.max_len = max_len

        # token + positional embeddings
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_len, embed_size)

        # attention pre-norm
        self.ln1 = nn.LayerNorm(embed_size)
        # attention projections
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # feedforward MLP
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),  # expansion
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(dropout),
        )

        # final LayerNorm before output
        self.lnf = nn.LayerNorm(embed_size)
        # output projection
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        B, T = x.shape

        # token + positional embeddings
        tok_emb = self.embed(x)                      # [B, T, E]
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pos_embed(pos)[None, :, :]    # [1, T, E]
        x = tok_emb + pos_emb                        # [B, T, E]

        # --- Attention block ---
        x_norm = self.ln1(x)
        Q = self.query(x_norm)
        K = self.key(x_norm)
        V = self.value(x_norm)

        att = (Q @ K.transpose(-2, -1)) / (self.embed_size ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        context = self.attn_dropout(att @ V)
        x = x + context  # residual

        # --- MLP block ---
        x_norm2 = self.ln2(x)
        x = x + self.mlp(x_norm2)  # residual

        # --- final normalization and output ---
        x = self.lnf(x)
        logits = self.fc(x)  # [B, T, vocab_size]
        return logits
