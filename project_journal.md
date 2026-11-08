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

### 2026-11-08 — Downloaded: TinyStories and Alpaca-est (not used yet)
- `data/raw/tinystories/`: TinyStoriesV2-GPT4 train (2.23 GB, ~2.7M stories) and valid (22.5 MB,
  27,630 stories) from Hugging Face `roneneldan/TinyStories`, licence CDLA-Sharing-1.0. Very simple
  English stories, separated by `<|endoftext|>`, median 719 chars (~1 window). A few cp1252 control
  chars (-) to fix; the train file ends mid-story.
- `data/raw/alpaca-est/`: TartuNLP Alpaca-est (`alpaca_est.json`, 28.9 MB, 52,006 Estonian
  instruction/input/output examples, generated with gpt-3.5-turbo; 17.5k have an input). Median
  instruction 60 chars, output 281 chars: one example fits a 256-token window. **No licence** in the
  repo: private use only, don't redistribute. Also the seed tasks and generation prompt.
- For prompt/response training, the loss should later be computed on the response tokens only.

**Naming convention (from 2026-11-08):** vocab sizes are total tokens, as is standard
(256 byte tokens + merges). `vocab-est-768.tsv` = 768 tokens = 512 merges. Earlier entries were
converted; model folder names like `v512-...` keep the old merge-count names.

### 2026-11-08 — Vocab files renamed to token counts; vocab 1256
- Renamed `vocab-*-0/256/512` -> `-256/512/768`, token caches too, and the vocab path inside all
  66 checkpoints (each verified: embedding rows = vocab size). `build_vocab.py` already named
  output as base + added, so it now counts tokens. New `s_train.py --vocab N` (tokens, default
  the largest) and `--embed N`.
- `vocab-{eng,est}-1256.tsv`: 488 more merges (1000 in total). eng 2.58 bytes/token (768: 2.28),
  est 2.39 (768: 2.10). Longest tokens: ` politician`, ` President`, ` Nõukogude`, ` kirjanik`.

### 2026-11-08 — Final 20k-step models (`models/final-20k/`)
- Recipe from all experiments so far: plain Wikipedia training, 2 layers x 2 heads, embed 64,
  vocab 768 tokens (512 merges), 215k params, 20k steps, seed 1. ~26 min each on the GTX 980.

| lang | val loss | bits/byte | vs 10k run (v512-2x64-2h) | train loss |
|---|---|---|---|---|
| est | 3.358 | **2.331** | 2.505 (−0.17) | 3.369 |
| eng | 3.272 | **2.045** | 2.196 (−0.15) | 3.341 |

- Best checkpoint is the last step for both: still improving at 20k. No overfitting even though
  est saw its data ~120 times (train ≈ val).
- est aabits held-out: 3.104 (Wikipedia-only 10k: 3.192).
- Samples now have longer stretches of real words and grammar, still little meaning:
  "However, the city couple of the law of life is nothing too, but in the same same time to philosophs."
- Chat: `python sumerute.py eng --dir models/final-20k` (or `est`).
- Next ideas: even longer runs (the curve hasn't flattened); bigger model (4 layers / embed 128)
  now that training length is known to matter; aabits fine-tune (D recipe) on the est 20k model
  for simple Estonian.

### 2026-11-08 — Controls for B-mem: it's mostly extra training, not the aabits
- Same setup (est, 2x2, vocab 768), aabits held-out scored in every run. Bits per byte:

| run | steps | wiki | aabits held-out |
|---|---|---|---|
| A2 | 10k Wikipedia | 2.516 | 3.192 |
| B-mem | 5k aabits (memorized) + 10k Wikipedia | 2.425 | 3.066 |
| B-mem seed 2 | same, other seed | 2.423 | 3.069 |
| Ctrl-wikimem | 5k memorized Wikipedia slice (same size as aabits) + 10k Wikipedia | 2.404 | 3.120 |
| Ctrl-15k | 15k Wikipedia | **2.387** | 3.145 |

- B-mem reproduces (seed 2: 2.423), so it wasn't luck.
- But memorizing an equally small *Wikipedia* slice helps more, and spending all 15k steps on
  Wikipedia helps most. The B-mem gain is mostly "more optimizer steps", not simple text first.
  Any small memorized text is a useful warm-up compared with no warm-up; real data is better.
- aabits first leaves a small edge on simple text (3.07 vs 3.15); fine-tuning at the end (D, 2.63)
  is far better for that.
- Big side finding: **the models are undertrained and the LR schedule matters.** At step 10k,
  Ctrl-15k (lr still 3.3e-4) was at 3.497 val loss vs A2 3.624 (lr already decayed to 1e-4).
  10k-step runs anneal too early; longer runs at high LR keep improving.
- Corrected conclusion: curriculum with aabits doesn't beat the same steps on Wikipedia.
  Next: 20k-step runs for both languages.

### 2026-11-08 — B-mem: memorize aabits first, then Wikipedia (surprise)
- Phase 1: aabits only, 5000 steps (train loss 4.5 -> 0.36; held-out stories 3.11 -> 7.04 bits/byte,
  i.e. memorized, not generalized). Phase 2: the last (memorized) weights, then 10k Wikipedia
  steps with the same settings as B. Folders `models/exp-aabits/Bmem1`, `Bmem2`.

| step | A2 (no aabits) | B (aabits 280 steps) | B-mem (aabits 5000 steps) |
|---|---|---|---|
| 250 | 5.211 | 4.756 | 4.806 |
| 1000 | 4.247 | 4.212 | 4.174 |
| 2000 | 4.077 | 4.065 | 3.887 |
| 5000 | 3.809 | 3.784 | 3.589 |
| 10000 | 3.624 | 3.626 | **3.493** |
| final wiki bits/byte | 2.516 | 2.517 | **2.425** |
| final aabits held-out | 3.192 | 3.191 | 3.066 |

- −0.09 bits/byte on Wikipedia, ~9x the measured run-to-run noise, and the gap is still there at the
  end. Contradicts the earlier "curriculum doesn't help" conclusion — the length of the first phase
  matters. The user's intuition may be right.
- Not yet ruled out: (1) B-mem simply had 5000 more optimizer steps in total; (2) it's not about
  *simple* text but about memorizing any small text first; (3) one lucky run. Controls needed
  before believing it.

### 2026-11-08 — Experiment: where to put the aabits (curriculum vs mixing vs fine-tuning)
- All est, 2 layers x 2 heads, vocab 768. Scored on Wikipedia val and on 5 held-out aabits
  stories (every 10th, never trained on). Bits per byte, lower is better.

| run | what | wiki | aabits held-out |
|---|---|---|---|
| A | Wikipedia only (earlier run) | **2.505** | 3.190 |
| A2 | Wikipedia only, other seed | 2.516 | 3.192 |
| B | aabits only 280 steps (best held-out), then Wikipedia 10k | 2.517 | 3.191 |
| C1 | Wikipedia + aabits x1 (0.3%) | 2.543 | 3.125 |
| C10 | Wikipedia + aabits x10 (3%) | 2.538 | 2.822 |
| D | A, then 1k steps Wikipedia + aabits x100 (23%), lr 3e-4 | 2.540 | **2.627** |

- **Run-to-run noise** (A vs A2): ~0.01 bits/byte. Differences below ~0.02 mean nothing.
- **Curriculum (B)**: a head start on Wikipedia that fades — wiki val loss after 250 steps
  4.76 vs 5.21 (A2), 1000: 4.21 vs 4.25, 2000: 4.07 vs 4.08, end: same. What it learned on
  aabits was fully forgotten (held-out 3.105 after the aabits phase -> 3.191 after Wikipedia,
  same as never seeing it). No lasting effect on either score.
- **Mixing (C)**: costs ~0.03 on Wikipedia, gains on simple text with the repeat factor.
- **Fine-tuning at the end (D)**: by far the best on simple text (−0.56), same small Wikipedia
  cost as mixing, and only 1k extra steps. Its samples pick up the book's style: quotes,
  characters (Teele, isa), sünnipäev.
- B's aabits phase did not reach memorization: the 215k-param model was at train loss 3.16
  after 600 steps (held-out best at step 280). A "train until memorized" run needs far more steps.
- Conclusion: simple text belongs at the end (or mixed in), not at the start.

### 2026-11-08 — No more 1-head models
- Decision: new models use 2+ heads and always have the attention output projection.
  `s_train.py` defaults to 2 layers x 2 heads; `--no-proj` removed. `s_model.py` still builds
  models without the projection, so old checkpoints (baselines, 1-head runs) keep loading.
- Not measured, and now won't be: how much of the 2-head gain came from the projection itself.

### 2026-11-08 — Simple Estonian: aabits
- No Simple Estonian Wikipedia exists, and no open children's-text corpus was found (ENC is mostly
  web text; the fiction corpus is adult). Source: a simplified first-reader PDF in `data/raw/`
  (Vaiksoo, Toomla, Ilves "Aabits", simplified by Kadrian Hiie). Copyrighted: private use only.
- `clean_aabits.py` -> `data/aabits-est.txt`: pdftotext, join compound hyphens (HEINA-MAA ->
  heinamaa) and line-end splits, all-caps -> sentence case with a hand-made name list (inflected
  forms via endings), capital after „ in direct speech, one paragraph + EOT per story.
- 57 stories, 5,214 words, 36 KB: ~0.3% of the est Wikipedia data. Mixed in once it would barely
  register; options are repeating it or a short final training phase on simple text.
- Source switched to `aabits_veelgi_lihtsam_.pdf`: the same text split into syllables
  (SU-VI ON TO-RE), but with typos fixed (vanaisa, garderoobis, paljud pered, alt, pintsak).
  Since all hyphens between letters are removed, the syllable split doesn't matter.
- `FIXES` in `clean_aabits.py` for typos in both PDFs: klegutama -> kelgutama, j aksa -> jaksa,
  soovides t aru -> soovidest aru, mixed dashes. Spaced hyphens become en dashes.

### 2026-11-08 — Vocab 768 tokens (512 merges)
- `vocab-{eng,est}-768.tsv` (named `-512` until the rename): 256 more merges on top of 256. eng 2.28 bytes/token (was 1.99),
  est 2.10 (was 1.85). A 256-token window now covers ~14% more text.
- Same 4 shapes as before, 10k steps; 1-head runs without projection, 2-head runs with it.
  Folders `models/v512-<layers>x64-<heads>h/` (folder names use the old merge count). Comparison in bits per byte:

| lang | layers x heads | bpb vocab 512 | bpb vocab 768 | change | params (768) |
|---|---|---|---|---|---|
| est | 1 x 1 | 2.757 | 2.643 | −0.114 | 161k |
| est | 1 x 2 | 2.716 | 2.614 | −0.102 | 165k |
| est | 2 x 1 | 2.655 | 2.520 | −0.135 | 207k |
| est | 2 x 2 | 2.554 | **2.505** | −0.049 | 215k |
| eng | 1 x 1 | 2.430 | 2.289 | −0.142 | 161k |
| eng | 1 x 2 | 2.417 | 2.277 | −0.140 | 165k |
| eng | 2 x 1 | 2.296 | 2.209 | −0.087 | 207k |
| eng | 2 x 2 | 2.304 | **2.196** | −0.107 | 215k |

- The bigger vocab helps every shape, by 0.05–0.14 bits/byte, for only ~5% more time per run.
  The extra 33k params are all in the embedding and output layers.
- With vocab 768, 2 heads help English a little (2.209 -> 2.196) and Estonian much less than
  before (2.520 -> 2.505). Differences this small still need the run-to-run noise measured.
- Still no overfitting (train ≈ val).

### 2026-11-07 — Multi-head attention (plan step 3c)
- `Block` splits Q/K/V into `n_heads` heads of `embed/n_heads` dims, each with its own attention
  pattern; optional output projection `attn_proj` mixes the heads. Old checkpoints load as
  1 head, no projection (identical outputs). New options: `s_train.py --layers --heads --no-proj --out`,
  `sumerute.py --dir`. Runs now live in folders: `models/<layers>x<embed>-<heads>h/`.
- All runs: embed 64, 512 vocab, 10k steps. 2-head runs include the projection, 1-head runs don't.

| lang | layers x heads | params | val loss | bits/byte | time |
|---|---|---|---|---|---|
| est | 1 x 1 | 128k | 3.504 | 2.757 | 6.2 min |
| est | 1 x 2 | 132k | 3.451 | 2.716 | 7.7 min |
| est | 2 x 1 | 174k | 3.374 | 2.655 | 10.0 min |
| est | 2 x 2 | 182k | **3.246** | **2.554** | 12.9 min |
| eng | 1 x 1 | 128k | 3.357 | 2.430 | 6.2 min |
| eng | 1 x 2 | 132k | 3.338 | 2.417 | 7.6 min |
| eng | 2 x 1 | 174k | **3.171** | **2.296** | 10.0 min |
| eng | 2 x 2 | 182k | 3.182 | 2.304 | 13.1 min |

- Estonian gains clearly from 2 heads (−0.05 at 1 layer, −0.13 at 2 layers); English barely
  (−0.02 at 1 layer, +0.01 at 2 layers, likely within run-to-run noise, which isn't measured yet).
  Possibly Estonian's long words with endings need two things tracked at once; not proven.
- Layers matter more than heads for English; for Estonian both help and add up.
- 2 heads cost ~25–30% more time per step on the GTX 980 (reshapes), for +3% params.
- Open: measure run-to-run noise (same config, different seed) before reading small differences.

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
- Result for the eng 512-token vocab (256 merges): 1.99 bytes/token (2.01 before), first merges ` t`, ` a`, `he`, `in`, ` the`;
  longest tokens ` American`, ` their`, ` which`. ". This is" -> `.` ` this`-style tokens.
- Old cross-word vocab kept as `vocab-eng-512-crossword.tsv` for comparison (not picked up by
  the scripts). Vocabs built before this change encode worse now, since their cross-word merges
  no longer apply: rebuild them (est too).
- est 512-token vocab rebuilt from the 256-token byte base: 1.85 bytes/token, 19 s. `ä` and `õ` became whole-character
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
- `vocab-eng-512.tsv` (then named `-256`): 512 tokens, 2.01 bytes/token. Larger vocabs (2k+) are now practical.

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
