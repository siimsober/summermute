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
from tokenizer import BPETokenizer

# ---------- Settings ----------
MODEL_CONFIG = dict(embed_size=64, max_len=256, dropout=0.1, n_layers=2, n_heads=1, attn_proj=True)
BATCH_SIZE = 128
LR = 1e-3              # peak learning rate
MIN_LR = 1e-4          # learning rate at the end of the cosine decay
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.1     # AdamW, applied to weight matrices only
GRAD_CLIP = 1.0
EVAL_EVERY = 250       # steps between validation runs and checkpoints
EVAL_BATCHES = 40      # fixed batches averaged for the validation loss
VAL_BLOCK = 16384      # validation is every VAL_EVERY-th block of this many tokens
VAL_EVERY = 20         # -> 5% of the tokens, spread across the whole file


# ---------- Vocab Finder ----------
def find_latest_vocab_number(vocab_dir="data/vocab", lang="eng"):
    """
    Find the largest number XXX in vocab-{lang}-XXX.tsv files.
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
    cache = f"data/tokens-{vocab_name}.npy"
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


def get_batch(tokens, starts, context_len, device):
    """Windows of context_len tokens at the given start indices -> (x, y) shifted by one token."""
    idx = starts[:, None] + np.arange(context_len + 1)
    batch = torch.from_numpy(tokens[idx].astype(np.int64)).to(device, non_blocking=True)
    return batch[:, :-1], batch[:, 1:]


def lr_at(step, total_steps):
    """Linear warmup, then cosine decay from LR to MIN_LR."""
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = min(1.0, (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS))
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))


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
parser.add_argument("--out", default="models", help="folder for checkpoints and log (default: models)")
parser.add_argument("--layers", type=int, help=f"override n_layers (default {MODEL_CONFIG['n_layers']})")
parser.add_argument("--heads", type=int, help=f"override n_heads (default {MODEL_CONFIG['n_heads']})")
parser.add_argument("--no-proj", action="store_true", help="no output projection after attention")
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

model_config = dict(MODEL_CONFIG)
if args.layers:
    model_config["n_layers"] = args.layers
if args.heads:
    model_config["n_heads"] = args.heads
if args.no_proj:
    model_config["attn_proj"] = False

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load vocab, or the one the checkpoint was trained with ----------
checkpoint = None
if resume:
    checkpoint = torch.load(last_name, map_location=device)
    vocab_path = checkpoint["config"]["vocab_path"]
    print(f"✅ Resuming from {last_name} at step {checkpoint['step']}")
else:
    latest_num = find_latest_vocab_number("data/vocab", lang)
    vocab_path = f"data/vocab/vocab-{lang}-{latest_num}.tsv"

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
train_tokens, val_tokens = split_train_val(tokens)
context_len = config["max_len"]
token_bytes = np.array([len(tokenizer.id_to_bytes.get(i, ())) for i in range(vocab_size)])
val_bytes_per_token = token_bytes[val_tokens].mean()
print(f"✅ {len(train_tokens)} train / {len(val_tokens)} val tokens, "
      f"{val_bytes_per_token:.2f} bytes/token in val")

# Fixed validation windows, so val loss is comparable between evaluations and runs
rng = np.random.default_rng(0)
val_starts = rng.integers(0, len(val_tokens) - context_len - 1, size=(EVAL_BATCHES, BATCH_SIZE))
train_rng = np.random.default_rng()

# ---------- Optimizer ----------
decay = [p for p in model.parameters() if p.dim() >= 2]       # weight matrices, embeddings
no_decay = [p for p in model.parameters() if p.dim() < 2]     # biases, LayerNorm
optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": WEIGHT_DECAY},
    {"params": no_decay, "weight_decay": 0.0},
], lr=LR, betas=(0.9, 0.95))
loss_fn = nn.CrossEntropyLoss()

start_step = 0
best_val = float("inf")
if checkpoint:
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_step = checkpoint["step"]
    best_val = checkpoint["best_val"]

if not resume:
    with open(log_name, "w", newline="") as f:
        csv.writer(f).writerow(["step", "train_loss", "val_loss", "val_bits_per_byte", "lr", "seconds"])

# ---------- Training ----------
start_time = time.time()
print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))} on {device}")
print(f"Steps {start_step}..{total_steps}, batch {BATCH_SIZE} x {context_len} tokens")

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

        if step % EVAL_EVERY == 0 or step == total_steps:
            train_loss = running_loss / running_count
            running_loss, running_count = 0.0, 0
            val_loss = evaluate(model, val_tokens, val_starts, context_len, vocab_size, device)
            bpb = val_loss / math.log(2) / val_bytes_per_token
            elapsed = time.time() - start_time
            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model_name, step, best_val)
                marker = " ✅ best"
            save_checkpoint(last_name, step, best_val)
            print(f"Step {step}/{total_steps}: train {train_loss:.4f}, val {val_loss:.4f} "
                  f"({bpb:.3f} bits/byte), lr {lr:.2e}, {elapsed:.0f}s{marker}")
            with open(log_name, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{bpb:.4f}",
                                        f"{lr:.2e}", f"{elapsed:.0f}"])
except KeyboardInterrupt:
    save_checkpoint(last_name, step, best_val)
    print(f"\n⏸ Stopped at step {step}, saved {last_name}. Continue with --resume")
    sys.exit(0)

# ---------- Timing summary ----------
end_time = time.time()
duration = end_time - start_time
hours, rem = divmod(duration, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\n✅ Best val loss {best_val:.4f} saved to {model_name}")
print(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
print(f"⏱ Total duration: {int(hours)}h {int(minutes)}m {int(seconds)}s for {step - start_step} steps")
