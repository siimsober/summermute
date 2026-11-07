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

### 2026-11-07 — Experiment: 2 layers (before 3c)
- Only change: `n_layers=2` (embed 64, 1 head, 512 vocab). 173,824 params (+36%). 10k steps.

| lang | layers | val loss | val bits/byte | train loss | time |
|---|---|---|---|---|---|
| est | 1 (baseline) | 3.504 | 2.757 | 3.492 | 6.2 min |
| est | 2 | **3.374** | **2.655** | 3.390 | 10.0 min |
| eng | 1 (baseline) | 3.357 | 2.430 | 3.385 | 6.2 min |
| eng | 2 | **3.171** | **2.296** | 3.213 | 10.0 min |

- −0.13 (est) and −0.19 (eng) val loss, 4–6% fewer bits per byte: a much bigger gain than the
  parameter increase alone suggests. A second layer lets attention build on what the first found.
- Still no overfitting (train ≈ val; train is slightly higher because of dropout).
- English sample now has short grammatical stretches: "The friends were now borning the mathed
  it that it was dance".
- Baselines kept in `models/baseline-1x64/`; `models/sumerute-*.pt` are now the 2-layer models.

### 2026-11-07 — Causal mask as a buffer (plan step 3b)
- Each `Block` builds its `max_len x max_len` bool mask once (`register_buffer`, `persistent=False`)
  and slices `[:T, :T]`, instead of building a float mask every forward pass.
- Same outputs as 3a with the same weights (checked for T = 256, 17, 1); checkpoints unchanged,
  since non-persistent buffers aren't saved.
- Speed: training step 33.9 -> 31.9 ms (~6%, batch 128 x 256 on GTX 980); chat forward unchanged.

### 2026-11-07 — Block class (plan step 3a)
- Attention + MLP moved into `Block` in `s_model.py`; `myNN` holds `blocks = ModuleList` of
  `n_layers` blocks (new config key, default 1). No change to the math.
- Verified: same 128,192 params, and the old model and the new one with the same weights give
  identical outputs (max difference 0.0). 2 layers would be 173,824 params.
- Weight names changed (`key.weight` -> `blocks.0.key.weight`), so the baseline checkpoints were
  converted in place instead of retrained; originals in `models/pre-3a/`. Resume from a converted
  checkpoint continues at the baseline val loss.
- Step 3 is done in small steps, one change at a time, each compared against the baselines.

### 2026-11-07 — Better generation (plan step 2)
- `sumerute.py <lang> <temperature> [--top-k 40] [--top-p 0.95]`: top-k and top-p (nucleus)
  filtering before sampling, on by default; `--top-k 0 --top-p 1` turns them off.
- Output shows the prompt with its continuation, the token count and why it stopped
  (EOT or the 200-token limit). Paragraph stop was left out on purpose.
- Effect on the baseline: fewer junk tokens, but the model itself is the limit; word-level
  quality needs plan step 3.

### 2026-11-07 — Measurable training + baselines (plan step 1)
- `s_train.py <file> <steps> [--resume]`: random windows per step, AdamW (decay on weight matrices
  only, betas 0.9/0.95), 200-step warmup + cosine to 1e-4, grad clip 1.0, batch 128 x 256.
- Validation: every 20th block of 16k tokens (5%, spread over the file); 40 fixed batches, so
  val loss is comparable between runs. Reports bits per byte (comparable across vocab sizes).
- Checkpoints: `models/sumerute-{lang}.pt` = best val (used by `sumerute.py`),
  `-last.pt` = latest (resume, also written on Ctrl+C), `-log.csv` = one row per eval.
  Checkpoints hold the config incl. vocab path; `sumerute.py` uses it.
- Tokens cached in `data/tokens-vocab-{lang}-{n}.npy`, re-encoded when data or vocab is newer.
- Old pre-EOT models moved to `models/pre-eot/`.

**Baselines** (1 block, 1 head, embed 64, 512 vocab, 128k params, 10k steps, ~6 min on GTX 980):

| lang | train loss | val loss | val bits/byte |
|---|---|---|---|
| est | 3.492 | 3.504 | 2.757 |
| eng | 3.385 | 3.357 | 2.430 |

- Train ≈ val: no overfitting, the model is capacity-limited. Still improving slowly at 10k steps.
- Samples: English is word-like with real short words and broken long ones ("Abo the chountry was
  who was team."); Estonian stops early on EOT. Next: scale the model (plan step 3).

### 2026-11-07 — Pre-tokenize into words before BPE
- Problem: merges across word boundaries made tokens like `. T`, so ". This" became `. T` + `his `
  (the pronoun token) and every word had several spellings depending on what came before it.
- Fix, like GPT-2/GPT-4: `PRETOKENIZE` in `tokenizer.py` splits text into chunks (word with
  leading space, numbers of max 3 digits, punctuation, whitespace, contractions like `'s`, EOT alone)
  and merges never cross a chunk boundary, in both `build_vocab.py` and `encode()`.
- Result for eng 256: 1.99 bytes/token (2.01 before), first merges ` t`, ` a`, `he`, `in`, ` the`;
  longest tokens ` American`, ` their`, ` which`. ". This is" -> `.` ` this`-style tokens.
- Old cross-word vocab kept as `vocab-eng-256-crossword.tsv` for comparison (not picked up by
  the scripts). Vocabs built before this change encode worse now, since their cross-word merges
  no longer apply: rebuild them (est too).
- est 256 rebuilt from `vocab-est-0`: 1.85 bytes/token, 19 s. `ä` and `õ` became whole-character
  tokens within the first 10 merges; longest tokens ` aastal`, ` maakond`, ` mille`.
  Old est vocabs (1, 3, 128) moved to `data/vocab/old/`.

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
