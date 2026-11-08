# s_train.py
import sys
import argparse
import os
import re
import csv
import math
import time
import numpy as np
import torch
import torch.nn as nn
from s_model import myNN
from tokenizer import BPETokenizer, EOT_ID

# ---------- Settings ----------
# The attention output projection is always on for new models (old checkpoints without it still load)
MODEL_CONFIG = dict(embed_size=64, max_len=256, dropout=0.1, n_layers=2, n_heads=2, attn_proj=True)
BATCH_SIZE = 128
LR = 1e-3              # peak learning rate (--lr)
WARMUP_STEPS = 200     # (--warmup)
WEIGHT_DECAY = 0.1     # AdamW, applied to weight matrices only
GRAD_CLIP = 1.0
EVAL_EVERY = 250       # steps between validation runs and checkpoints (--eval-every)
EVAL_BATCHES = 40      # fixed batches averaged for the validation loss
VAL_BLOCK = 16384      # validation is every VAL_EVERY-th block of this many tokens
VAL_EVERY = 20         # -> 5% of the tokens, spread across the whole file
EXTRA_HOLDOUT_EVERY = 10   # every 10th story of the --extra file is held out for its own validation


# ---------- Vocab Finder ----------
def find_latest_vocab_number(vocab_dir="data/vocab", lang="eng"):
    """
    Find the largest number XXX in vocab-{lang}-XXX.tsv files (XXX = vocab size in tokens).
    If none exist, return 0.
    """
    pattern = re.compile(rf"vocab-{re.escape(lang)}-(\d+)\.tsv$")
    max_num = 0
    for fname in os.listdir(vocab_dir):
        match = pattern.match(fname)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num


# ---------- Token cache ----------
def load_tokens(train_file, vocab_path, tokenizer):
    """Encode the training file, or load the cached tokens if neither file changed since."""
    vocab_name = os.path.splitext(os.path.basename(vocab_path))[0]   # e.g. vocab-eng-256
    file_name = os.path.splitext(os.path.basename(train_file))[0]    # e.g. training-eng
    cache = f"data/tokens-{file_name}-{vocab_name}.npy"
    if os.path.exists(cache) and os.path.getmtime(cache) > max(os.path.getmtime(train_file),
                                                               os.path.getmtime(vocab_path)):
        print(f"✅ Loaded cached tokens from {cache}")
        return np.load(cache)
    with open(train_file, "rb") as f:
        data = f.read()
    print(f"✅ Read {len(data)} bytes from {train_file}, encoding...")
    tokens = tokenizer.encode(data).astype(np.uint16 if tokenizer.get_vocab_size() < 65536 else np.int32)
    np.save(cache, tokens)
    print(f"✅ Saved tokens to {cache}")
    return tokens


def split_train_val(tokens):
    """Every VAL_EVERY-th block of VAL_BLOCK tokens goes to validation."""
    block = np.arange(len(tokens)) // VAL_BLOCK
    is_val = block % VAL_EVERY == VAL_EVERY - 1
    return tokens[~is_val], tokens[is_val]


def split_stories(tokens):
    """Split a small file into stories (each ending with EOT); every EXTRA_HOLDOUT_EVERY-th is held out."""
    ends = np.flatnonzero(tokens == EOT_ID) + 1
    stories = np.split(tokens, ends)
    stories = [s for s in stories if len(s)]
    held = [s for i, s in enumerate(stories) if i % EXTRA_HOLDOUT_EVERY == EXTRA_HOLDOUT_EVERY - 1]
    train = [s for i, s in enumerate(stories) if i % EXTRA_HOLDOUT_EVERY != EXTRA_HOLDOUT_EVERY - 1]
    return np.concatenate(train), np.concatenate(held), len(stories), len(held)


def get_batch(tokens, starts, context_len, device):
    """Windows of context_len tokens at the given start indices -> (x, y) shifted by one token."""
    idx = starts[:, None] + np.arange(context_len + 1)
    batch = torch.from_numpy(tokens[idx].astype(np.int64)).to(device, non_blocking=True)
    return batch[:, :-1], batch[:, 1:]


def lr_at(step, total_steps):
    """Linear warmup, then cosine decay from peak to a tenth of it."""
    peak, low = args.lr, args.lr / 10
    if step < args.warmup:
        return peak * (step + 1) / args.warmup
    progress = min(1.0, (step - args.warmup) / max(1, total_steps - args.warmup))
    return low + 0.5 * (peak - low) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, val_tokens, val_starts, context_len, vocab_size, device):
    model.eval()
    losses = []
    for starts in val_starts:
        x, y = get_batch(val_tokens, starts, context_len, device)
        logits = model(x)
        losses.append(loss_fn(logits.view(-1, vocab_size), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate_text(model, tokens, context_len, device):
    """Mean loss over every token of a short text, read in consecutive windows."""
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(tokens) - 1, context_len):
        chunk = torch.from_numpy(tokens[i:i + context_len + 1].astype(np.int64)).to(device)[None, :]
        logits = model(chunk[:, :-1])
        total += nn.functional.cross_entropy(logits[0], chunk[0, 1:], reduction="sum").item()
        count += chunk.shape[1] - 1
    model.train()
    return total / count


def save_checkpoint(path, step, best_val):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "config": config,
    }, path)


# ---------- Command-line input ----------
parser = argparse.ArgumentParser(description="Train a sumerute model")
parser.add_argument("train_file", help="e.g. data/training-eng.txt")
parser.add_argument("steps", type=int, help="total training steps")
parser.add_argument("--resume", action="store_true", help="continue from <out>/sumerute-<lang>-last.pt")
parser.add_argument("--init", help="start from this checkpoint's weights, with a fresh optimizer and schedule")
parser.add_argument("--out", default="models", help="folder for checkpoints and log (default: models)")
parser.add_argument("--layers", type=int, help=f"override n_layers (default {MODEL_CONFIG['n_layers']})")
parser.add_argument("--heads", type=int, help=f"override n_heads (default {MODEL_CONFIG['n_heads']})")
parser.add_argument("--embed", type=int, help=f"override embed_size (default {MODEL_CONFIG['embed_size']})")
parser.add_argument("--vocab", type=int, help="vocab size in tokens, e.g. 768 for data/vocab/vocab-<lang>-768.tsv "
                                              "(default: the largest one)")
parser.add_argument("--lr", type=float, default=LR, help=f"peak learning rate (default {LR})")
parser.add_argument("--warmup", type=int, default=WARMUP_STEPS, help=f"warmup steps (default {WARMUP_STEPS})")
parser.add_argument("--eval-every", type=int, default=EVAL_EVERY, help=f"steps between evals (default {EVAL_EVERY})")
parser.add_argument("--seed", type=int, help="random seed for weights and batches")
parser.add_argument("--extra", help="small text file (stories ending with EOT) mixed into training; "
                                    f"every {EXTRA_HOLDOUT_EVERY}th story is held out and scored separately")
parser.add_argument("--extra-repeat", type=int, default=1, help="how many times the extra text is repeated")
parser.add_argument("--main-repeat", type=int, default=1, choices=[0, 1],
                    help="0 = train on the extra text only (main file is still used for validation)")
parser.add_argument("--best-on", default="main", choices=["main", "extra"],
                    help="which validation loss picks the best checkpoint")
args = parser.parse_args()

train_file = args.train_file
total_steps = args.steps
resume = args.resume
base = os.path.splitext(os.path.basename(train_file))[0]  # e.g. "training-eng"
lang = base.split('-')[-1]
model_name = f"{args.out}/sumerute-{lang}.pt"        # best validation loss, used by sumerute.py
last_name = f"{args.out}/sumerute-{lang}-last.pt"    # latest step, used to resume
log_name = f"{args.out}/sumerute-{lang}-log.csv"
os.makedirs(args.out, exist_ok=True)
if args.seed is not None:
    torch.manual_seed(args.seed)

model_config = dict(MODEL_CONFIG)
if args.layers:
    model_config["n_layers"] = args.layers
if args.heads:
    model_config["n_heads"] = args.heads
if args.embed:
    model_config["embed_size"] = args.embed

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load vocab, or the one the checkpoint was trained with ----------
checkpoint = None
if resume or args.init:
    checkpoint = torch.load(last_name if resume else args.init, map_location=device)
    vocab_path = checkpoint["config"]["vocab_path"]
    print(f"✅ {'Resuming from ' + last_name if resume else 'Starting from weights of ' + args.init} "
          f"(step {checkpoint['step']})")
else:
    vocab_num = args.vocab or find_latest_vocab_number("data/vocab", lang)
    vocab_path = f"data/vocab/vocab-{lang}-{vocab_num}.tsv"
if checkpoint and args.vocab:
    print(f"⚠️ --vocab ignored: the checkpoint's vocab {vocab_path} is used")

if not os.path.exists(vocab_path):
    print(f"❌ Error: No vocab file found at {vocab_path}")
    sys.exit(1)

tokenizer = BPETokenizer(vocab_path)
vocab_size = tokenizer.get_vocab_size()
print(f"Tokenizer initialized from {vocab_path} with {vocab_size} token vocabulary.")

config = checkpoint["config"] if checkpoint else dict(model_config, vocab_path=vocab_path, vocab_size=vocab_size)
model = myNN.from_config(vocab_size, config).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {config.get('n_layers', 1)} layers, {config.get('n_heads', 1)} heads, embed {config['embed_size']}, "
      f"projection {'on' if config.get('attn_proj') else 'off'}")
print(f"Total trainable parameters: {total_params}")

# ---------- Data ----------
tokens = load_tokens(train_file, vocab_path, tokenizer)
main_train, val_tokens = split_train_val(tokens)
context_len = config["max_len"]
token_bytes = np.array([len(tokenizer.id_to_bytes.get(i, ())) for i in range(vocab_size)])
val_bytes_per_token = token_bytes[val_tokens].mean()
print(f"✅ {len(main_train)} train / {len(val_tokens)} val tokens, "
      f"{val_bytes_per_token:.2f} bytes/token in val")

parts = [main_train] if args.main_repeat else []
extra_val = None
if args.extra:
    extra_train, extra_val, n_stories, n_held = split_stories(load_tokens(args.extra, vocab_path, tokenizer))
    extra_bytes_per_token = token_bytes[extra_val].mean()
    parts += [extra_train] * args.extra_repeat
    print(f"✅ Extra {args.extra}: {n_stories} stories, {n_held} held out ({len(extra_val)} tokens), "
          f"{len(extra_train)} train tokens x {args.extra_repeat}")
train_tokens = np.concatenate(parts) if parts else None
if train_tokens is not None and args.extra:
    share = len(extra_train) * args.extra_repeat / len(train_tokens)
    print(f"✅ Training on {len(train_tokens)} tokens, {share:.1%} of them from the extra text")

# Fixed validation windows, so val loss is comparable between evaluations and runs
rng = np.random.default_rng(0)
val_starts = rng.integers(0, len(val_tokens) - context_len - 1, size=(EVAL_BATCHES, BATCH_SIZE))
train_rng = np.random.default_rng(args.seed)

# ---------- Optimizer ----------
decay = [p for p in model.parameters() if p.dim() >= 2]       # weight matrices, embeddings
no_decay = [p for p in model.parameters() if p.dim() < 2]     # biases, LayerNorm
optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": WEIGHT_DECAY},
    {"params": no_decay, "weight_decay": 0.0},
], lr=args.lr, betas=(0.9, 0.95))
loss_fn = nn.CrossEntropyLoss()

start_step = 0
best_val = float("inf")
if checkpoint:
    model.load_state_dict(checkpoint["model"])
if resume:
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_step = checkpoint["step"]
    best_val = checkpoint["best_val"]

LOG_COLUMNS = ["step", "train_loss", "val_loss", "val_bits_per_byte", "extra_val_loss", "extra_bits_per_byte",
               "lr", "seconds"]
if not resume:
    with open(log_name, "w", newline="") as f:
        csv.writer(f).writerow(LOG_COLUMNS)


def eval_and_log(step, train_loss, lr):
    """Validate, save checkpoints, print and log one row. Returns nothing; updates best_val."""
    global best_val
    val_loss = evaluate(model, val_tokens, val_starts, context_len, vocab_size, device)
    bpb = val_loss / math.log(2) / val_bytes_per_token
    extra_loss = extra_bpb = None
    if extra_val is not None:
        extra_loss = evaluate_text(model, extra_val, context_len, device)
        extra_bpb = extra_loss / math.log(2) / extra_bytes_per_token
    score = extra_loss if args.best_on == "extra" else val_loss
    elapsed = time.time() - start_time
    marker = ""
    if score < best_val and step > start_step:
        best_val = score
        save_checkpoint(model_name, step, best_val)
        marker = " ✅ best"
    if step > start_step:
        save_checkpoint(last_name, step, best_val)
    extra_text = f", extra {extra_loss:.4f} ({extra_bpb:.3f} bits/byte)" if extra_loss is not None else ""
    train_text = f"train {train_loss:.4f}, " if train_loss is not None else ""
    print(f"Step {step}/{total_steps}: {train_text}val {val_loss:.4f} ({bpb:.3f} bits/byte){extra_text}, "
          f"lr {lr:.2e}, {elapsed:.0f}s{marker}")
    fmt = lambda v, spec: format(v, spec) if v is not None else ""
    with open(log_name, "a", newline="") as f:
        csv.writer(f).writerow([step, fmt(train_loss, ".4f"), f"{val_loss:.4f}", f"{bpb:.4f}",
                                fmt(extra_loss, ".4f"), fmt(extra_bpb, ".4f"), f"{lr:.2e}", f"{elapsed:.0f}"])


# ---------- Training ----------
start_time = time.time()
print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))} on {device}")
print(f"Steps {start_step}..{total_steps}, batch {BATCH_SIZE} x {context_len} tokens, peak lr {args.lr}")

if args.init:
    eval_and_log(0, None, 0.0)   # score of the starting weights

step = start_step
running_loss, running_count = 0.0, 0
try:
    while step < total_steps:
        lr = lr_at(step, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        starts = train_rng.integers(0, len(train_tokens) - context_len - 1, size=BATCH_SIZE)
        x, y = get_batch(train_tokens, starts, context_len, device)
        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        step += 1
        running_loss += loss.item()
        running_count += 1

        if step % args.eval_every == 0 or step == total_steps:
            eval_and_log(step, running_loss / running_count, lr)
            running_loss, running_count = 0.0, 0
except KeyboardInterrupt:
    save_checkpoint(last_name, step, best_val)
    print(f"\n⏸ Stopped at step {step}, saved {last_name}. Continue with --resume")
    sys.exit(0)

# ---------- Timing summary ----------
end_time = time.time()
duration = end_time - start_time
hours, rem = divmod(duration, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\n✅ Best {args.best_on} val loss {best_val:.4f} saved to {model_name}")
print(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
print(f"⏱ Total duration: {int(hours)}h {int(minutes)}m {int(seconds)}s for {step - start_step} steps")
