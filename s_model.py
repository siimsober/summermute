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

        # token + positional embeddings
        self.embed = nn.Embedding(vocab_size, self.embed_size)
        self.pos_embed = nn.Embedding(self.max_len, self.embed_size)

        # layer norm before attention (pre-norm)
        self.ln1 = nn.LayerNorm(self.embed_size)

        # attention projections
        self.key = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.query = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.value = nn.Linear(self.embed_size, self.embed_size, bias=False)

        self.fc = nn.Linear(self.embed_size, vocab_size)
        self.lnf = nn.LayerNorm(self.embed_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T = x.shape

        # token + positional embeddings
        tok_emb = self.embed(x)                      # [B, T, E]
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pos_embed(pos)[None, :, :]    # [1, T, E]
        x = tok_emb + pos_emb                        # [B, T, E]

        # apply layernorm before attention
        x_norm = self.ln1(x)
        # compute Q, K, V
        Q = self.query(x_norm)                            # [B, T, E]
        K = self.key(x_norm)                              # [B, T, E]
        V = self.value(x_norm)                            # [B, T, E]

        # scaled dot-product attention
        att = (Q @ K.transpose(-2, -1)) / (self.embed_size ** 0.5)  # [B, T, T]
        
        # causal mask to only fuse preceding tokens
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)                 # [B, T, T]
        context = self.dropout(att @ V)                            # [B, T, E]

        x = x + context
        x = self.lnf(x)
        x = self.dropout(x)
        logits = self.fc(x)                    # [B, T, vocab_size]
        return logits
    