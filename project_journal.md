# Sumerute project journal

A small language model built from scratch to learn how LMs work, trained on English
(Project Gutenberg books) and Estonian (Wikipedia) text.

## Current state

- **Model** (`s_model.py`): 1 transformer block, single-head causal attention, 4x GELU MLP,
  pre-LayerNorm, learned positional embeddings. embed=64, context=256, ~110k params.
- **Tokenizer** (`tokenizer.py`, `build_vocab.py`): byte-level BPE-style vocab (256 bytes + merges),
  greedy longest-match encoding (max 10 bytes per token). eng: 100 merges, est: 128 merges.
- **Training** (`s_train.py`): Adam lr=1e-3, batch 600, every-position sliding window, saves only at end.
- **Inference** (`sumerute.py`): temperature sampling over the last `max_len` tokens.
- **Data**: `data/training-eng.txt` (from `clean_gutenberg.py`), `data/training-est.txt`
  (from `clean_wikipedia.py`). `data/` and `models/` are gitignored.

## Decisions log

Newest first. Record what changed, why, and what was learned.

### 2026-11-07 — Data cleaning fixes
- Gutenberg: strip header/footer for both `THIS` and `THE PROJECT GUTENBERG EBOOK` marker variants
  (22 of 40 books were keeping the license boilerplate), drop the marker line itself, cut the
  license at the END marker.
- Wikipedia: parse the XML properly and keep only article text (main namespace, no redirects)
  instead of regex-stripping tags, which left page ids, timestamps, usernames and edit
  summaries in the training text. Also remove `<ref>` content, nested templates, tables,
  categories and wiki markup.
- Why: the model was spending capacity memorizing metadata and boilerplate.

### 2026-11-07 — Assessment
Priorities agreed, in order:
1. Clean the data (above).
2. Add a validation split, step-based training with random windows, periodic checkpoints.
3. Bigger vocab (2k–8k merges) with a faster merge loop.
4. Scale the model: multiple layers/heads, attention output projection, weight tying,
   AdamW + warmup/cosine LR, grad clipping, mixed precision.

### 2025-11 — Feedforward block + generation fix
- Added the MLP block with its own pre-LayerNorm and residual; moved dropout into blocks.
- Generation previously fed one token at a time (no context); now feeds the rolling last
  `max_len` tokens.

### 2025-10/11 — Earlier history (see git log and `archive/`)
- v1–v8: from simple byte models with fixed 2/4-byte context, through embeddings, an RNN,
  mock attention, to real single-head attention with LayerNorm and dropout.
- Introduced the byte-pair tokenizer and separate eng/est vocabs and models.

## Open questions / ideas
- Use one shared vocab for both languages, or keep them separate?
- Is ~12 MB per language enough, or pull more Gutenberg books / more of etwiki?
