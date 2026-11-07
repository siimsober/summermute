# Sumerute plan

What to do next, in order. Decisions and results go in `project_journal.md`.

## Where we are

- Data: clean eng (Simple Wikipedia + Gutenberg, 23.5 MB) and est (Wikipedia, 12 MB), with EOT segment markers.
- Tokenizer: fast BPE, merges stay inside words (GPT-2 style), 256 merges per language.
- Model: 1 block, single-head attention, embed 64, context 256, ~110k params.
- Training: no validation, "epoch" = every position (huge), saves only at the end.

The biggest gap now is training: there is no way to measure whether a change helped.
Step 1 fixes that, and every later step is judged by its numbers.

## 1. Training you can measure

Goal: a baseline validation loss for the current model, for both languages.

- [x] **Cache encoded tokens**: save `data/tokens-{lang}-{vocab}.npy` after encoding, reuse if the
      vocab hasn't changed (encoding 23.5 MB takes ~30 s, more with bigger vocabs).
- [x] **Validation split**: hold out ~5% spread across the file (e.g. every 20th block of 64k tokens),
      not the end. The end of the eng file is all Gutenberg.
- [x] **Step-based training**: random windows of `context_len` tokens per batch, train for N steps
      instead of epochs.
- [x] **Evaluate every N steps**: mean val loss over a fixed set of batches, printed with train loss.
- [x] **Report bits per byte** too: `loss / ln(2) * tokens / bytes`. Token-level loss can't be
      compared across vocab sizes, bits per byte can.
- [x] **Checkpoints**: save model, optimizer, step, config (vocab path, sizes) every N steps and
      on best val loss. Ctrl+C keeps the latest one, and training can resume.
- [x] **Optimizer**: AdamW, short warmup then cosine decay, gradient clipping at 1.0.
- [x] **Log to CSV** (step, train loss, val loss, lr, time) so runs can be plotted and compared.
- [x] **`sumerute.py` loads the config from the checkpoint** instead of relying on matching
      defaults, and uses `map_location` so a GPU-trained model loads on CPU.

Done when: the journal has baseline val loss / bits per byte for eng and est with the current model.

## 2. Better generation

Cheap and makes testing models nicer.

- [ ] **top-k / top-p sampling** alongside temperature.
- [ ] **Stop options**: on EOT (done), or at the end of a paragraph (`\n`).
- [ ] **Print the prompt with the output**, and how many tokens were generated and why it stopped.

## 3. Scale the model

Goal: lower val loss. Change one thing at a time and record each result.

- [ ] **`Block` class** (attention + MLP), stack `n_layers` of them.
- [ ] **Multi-head attention** with an output projection after the heads.
- [ ] **Weight tying**: output layer shares weights with the token embedding.
- [ ] **Mask as `register_buffer`** instead of rebuilding it every forward pass.
- [ ] **Mixed precision** on GPU (`torch.autocast`) for speed.
- [ ] **Sizes to try** (params roughly, with a 512 vocab):

  | layers | heads | embed | params |
  |---|---|---|---|
  | 1 | 1 | 64 | ~110k (baseline) |
  | 2 | 4 | 128 | ~0.5M |
  | 4 | 4 | 256 | ~3.3M |
  | 6 | 8 | 384 | ~11M |

  Rule of thumb: ~20 training tokens per parameter. 3M params wants ~60M tokens, more than the
  data has now, so step 5 comes in as the model grows.

## 4. Vocab size

- [ ] Train the same model with 256, 1k and 4k merges per language, compare **bits per byte**.
      Estonian (long words, many endings) will likely gain more from a bigger vocab than English.

## 5. More and broader data

- [ ] **Sample articles across the whole dump** (e.g. every Nth article) instead of the first ones,
      for broader topics at the same size.
- [ ] **Raise `MAX_BYTES`** when the model needs more tokens (see the rule of thumb above).
- [ ] Keep data cleaning scripts reproducible; note every data change in the journal.

## 6. Housekeeping

Any time, small.

- [ ] **`README.md`**: what the project is, how to run cleaning → vocab → training → chat.
- [ ] **`requirements.txt`**: torch, numpy, chardet.
- [ ] **One `find_latest_vocab_number`**: it is copied in three scripts, move it to `tokenizer.py`.
- [ ] **Remove duplicate virtualenvs** (`.venv`, `.venv313`, `venv313`), keep one.
- [ ] **`archive/`**: git history already keeps old versions; tag them and delete the folder.
- [ ] **Delete leftovers**: `data/vocab/vocab-eng-256-crossword.tsv`, `data/vocab/old/`, old `models/summermute-*.pt`.
