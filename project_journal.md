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

### 2026-11-07 — Fast, textbook BPE
- `encode()` now applies the merges in the order they were learned (textbook BPE) instead of
  greedy longest-match, so encoding matches how the vocab was trained. No 10-byte token limit.
- Vocab TSVs have a 4th `merge` column (`a,b` = the two token ids merged). Old files without it
  still load; the pair is worked out from the bytes.
- `build_vocab.py` counts pairs once, then each merge only updates the counts next to the merged
  positions (numpy, `merge_pair` in `tokenizer.py`). Checked against a full recount for 300 merges.
- Speed: 256 merges on 23.5 MB eng took 27 s (old version: ~45 s per merge, ~3 h).
- `vocab-eng-256.tsv`: 512 tokens, 2.01 bytes/token. Larger vocabs (2k+) are now practical.

### 2026-11-07 — English = Simple Wikipedia + Gutenberg
- `data/training-eng.txt` is now Simple Wikipedia (12 MB) followed by Gutenberg (11.5 MB),
  23.5 MB and 8,657 EOT segments. Rebuild with:
  `python clean_wikipedia.py eng`, then
  `cat data/training-eng.txt data/training-eng-gutenberg.txt > merged && mv merged data/training-eng.txt`.
  (`clean_gutenberg.py` writes to `training-eng.txt`; rename its output to `training-eng-gutenberg.txt` first.)
- Old vocabs to be deleted and rebuilt on this data.
- Caveat: a validation split taken from the end of the file would be all Gutenberg. Sample
  validation windows from across the file instead.

### 2026-11-07 — English data switched to Simple English Wikipedia
- `python clean_wikipedia.py eng|est`. English source is the Simple English Wikipedia dump
  (`data/raw/simplewiki/simplewiki-latest-pages-articles.xml.bz2`, 360 MB, dump of 2026-11-02),
  read straight from the .bz2.
- Why: short sentences and common words suit a tiny model; same format and EOT density as the
  Estonian data, so both models get comparable training text. Gutenberg chapters (~8k tokens)
  were too long for the model to learn to stop.
- Result: 12 MB, 2,642 articles, 8,244 segments (median ~825 chars). Only a few stray markup lines.
- Gutenberg is still available: `clean_gutenberg.py` writes `data/training-eng.txt` too; the
  last Gutenberg output was kept as `data/training-eng-gutenberg.txt`.
- The 12 MB cap uses only a small part of the dump. Raise `MAX_BYTES` when the model grows.

### 2026-11-07 — EOT markers for variable-length output
- Byte `0x04` (EOT, token id 4, `EOT_BYTE`/`EOT_ID` in `tokenizer.py`) marks the end of a text
  segment. It goes at the start of the line that begins the next segment (`...\n\x04CHAPTER II`).
- Where: after every book and article, before chapter headings (Gutenberg: CHAPTER/BOOK/PART/ACT/...
  or roman numerals) and before top-level `==` sections (Wikipedia). Replaces `=== NEW BOOK ===`.
- Minimum gap between EOTs (`MIN_CHAPTER` = 2000 chars, `MIN_SECTION` = 300) so table-of-contents
  lines and empty sections don't each get one.
- Not at every paragraph: that would teach "the next paragraph is unrelated" and would just
  duplicate the `\n` signal. Stopping at the end of a paragraph can be done at generation time instead.
- `build_vocab.py` never merges EOT with neighbours; `sumerute.py` stops generating on it.
- Result: eng 413 segments (median ~15k chars, chapters are long), est 7,359 segments
  (median ~840 chars).
- Also strips more transcriber credit lines ("Transcribed from…", "Scanned by…", "This etext was prepared…").

### 2026-11-07 — Data cleaning fixes
- Gutenberg: strip header/footer for both `THIS` and `THE PROJECT GUTENBERG EBOOK` marker variants
  (22 of 40 books were keeping the license boilerplate), drop the marker line itself, cut the
  license at the END marker.
  Old pre-2000s etexts use a "Small Print" header and "End of (Project Gutenberg's) Etext"
  footer instead, so those are handled too. "Produced by ..." credits are dropped.
- Wikipedia: stream the XML with `ElementTree.iterparse` and keep only article text
  (main namespace, no redirects or disambiguation pages) instead of regex-stripping tags,
  which left page ids, timestamps, usernames and edit summaries in the training text.
  Also removes `<ref>` content, nested templates, tables, file/category/language links,
  bold/italic quotes, list markers, and the tail sections (Viited, Välislingid, Kirjandus, ...).
  Stops at 12 MB of clean text (`MAX_BYTES`), about 2,000 articles.
- Why: the model was spending capacity memorizing metadata and boilerplate.
- Results: eng 11.5 MB / 40 books (10084.txt was wrongly skipped before), no license text
  left. est 12.0 MB, fewer than 50 of 103k lines with leftover markup.
- Known leftovers: 1249.txt (Anthem) is a comparison edition that embeds a second Gutenberg
  header in the body; a few transcriber notes mention Project Gutenberg.
- Consequence: existing vocabs and models were built from the old data and should be rebuilt.

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
