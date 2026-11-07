# s_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """One transformer block: attention (tokens gather from earlier tokens), then MLP (per token)."""
    def __init__(self, embed_size, max_len, dropout, n_heads=1, attn_proj=False):
        super().__init__()
        assert embed_size % n_heads == 0, "embed_size must be divisible by n_heads"
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.head_size = embed_size // n_heads

        # causal mask: position i may only look at positions <= i. Built once, moves with .to(device),
        # not saved in checkpoints (persistent=False)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)),
                             persistent=False)

        # attention pre-norm
        self.ln1 = nn.LayerNorm(embed_size)
        # attention projections
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)
        # mixes the heads' results back together (optional, older checkpoints don't have it)
        self.proj = nn.Linear(embed_size, embed_size) if attn_proj else None
        self.attn_dropout = nn.Dropout(dropout)

        # feedforward MLP
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),  # expansion
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, T, E = x.shape

        # --- Attention ---
        x_norm = self.ln1(x)
        # split E into n_heads heads of head_size each: [B, T, E] -> [B, H, T, head_size]
        H, D = self.n_heads, self.head_size
        Q = self.query(x_norm).view(B, T, H, D).transpose(1, 2)
        K = self.key(x_norm).view(B, T, H, D).transpose(1, 2)
        V = self.value(x_norm).view(B, T, H, D).transpose(1, 2)

        # every head computes its own attention pattern
        att = (Q @ K.transpose(-2, -1)) / (D ** 0.5)          # [B, H, T, T]
        att = att.masked_fill(~self.mask[:T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)

        # join the heads back together: [B, H, T, D] -> [B, T, E]
        context = (att @ V).transpose(1, 2).reshape(B, T, E)
        if self.proj is not None:
            context = self.proj(context)
        context = self.attn_dropout(context)
        x = x + context  # residual

        # --- MLP ---
        x_norm2 = self.ln2(x)
        x = x + self.mlp(x_norm2)  # residual
        return x


class myNN(nn.Module):
    def __init__(self, vocab_size, embed_size=64, max_len=256, dropout=0.1, n_layers=1,
                 n_heads=1, attn_proj=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.max_len = max_len

        # token + positional embeddings
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_len, embed_size)

        # stack of transformer blocks
        self.blocks = nn.ModuleList([Block(embed_size, max_len, dropout, n_heads, attn_proj)
                                     for _ in range(n_layers)])

        # final LayerNorm before output
        self.lnf = nn.LayerNorm(embed_size)
        # output projection
        self.fc = nn.Linear(embed_size, vocab_size)

    @classmethod
    def from_config(cls, vocab_size, config):
        """Build the model from a checkpoint config; keys missing in older checkpoints get old defaults."""
        return cls(vocab_size, config["embed_size"], config["max_len"], config["dropout"],
                   config.get("n_layers", 1), config.get("n_heads", 1), config.get("attn_proj", False))

    def forward(self, x):
        B, T = x.shape

        # token + positional embeddings
        tok_emb = self.embed(x)                      # [B, T, E]
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pos_embed(pos)[None, :, :]    # [1, T, E]
        x = tok_emb + pos_emb                        # [B, T, E]

        for block in self.blocks:
            x = block(x)                             # [B, T, E]

        # --- final normalization and output ---
        x = self.lnf(x)
        logits = self.fc(x)  # [B, T, vocab_size]
        return logits
