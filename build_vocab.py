import sys
import os
import re
import time
import numpy as np
from tokenizer import BPETokenizer, EOT_ID, merge_pair

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


# ---------- Pair counting ----------
# A pair (a, b) is stored as one int64 code a * PAIR_BASE + b
PAIR_BASE = 1 << 20

def count_pairs(tokens, starts):
    """Count the pairs (tokens[s], tokens[s+1]) for the given start indices. EOT pairs are skipped."""
    left = tokens[starts]
    right = tokens[starts + 1]
    ok = (left != EOT_ID) & (right != EOT_ID)   # EOT must stay a single token
    codes, counts = np.unique(left[ok] * PAIR_BASE + right[ok], return_counts=True)
    return zip(codes.tolist(), counts.tolist())

def neighbour_starts(positions, n):
    """Start indices of all pairs touching the given positions, within an array of length n."""
    starts = np.unique(np.concatenate(positions))
    return starts[(starts >= 0) & (starts <= n - 2)]


if len(sys.argv) < 3:
    print("Usage: python build_vocab.py <est|eng> <number of tokens to add>")
    sys.exit(1)

lang = sys.argv[1]
new_tokens = int(sys.argv[2])

# ---------- Config ----------
INPUT_FILE = f"data/training-{lang}.txt"
# ---------- Determine input vocab ----------
latest_vocab = find_latest_vocab_number("data/vocab", lang)
VOCAB_FILE = f"data/vocab/vocab-{lang}-{latest_vocab}.tsv"

if not os.path.exists(VOCAB_FILE):
    print(f"❌ Input vocab file not found: {VOCAB_FILE}")
    sys.exit(1)

VOCAB_OUT = f"data/vocab/vocab-{lang}-{latest_vocab + new_tokens}.tsv"

# ---------- Load tokenizer ----------
tokenizer = BPETokenizer(VOCAB_FILE)
print(f"✅ Tokenizer initialized with {VOCAB_FILE}")

# ---------- Read training data ----------
with open(INPUT_FILE, "rb") as f:
    data = f.read()
print(f"✅ Read {len(data)} bytes from {INPUT_FILE}")

start_time = time.time()
tokens = tokenizer.encode(data)
print(f"✅ Encoded with existing vocab: {len(tokens)} tokens ({len(data) / len(tokens):.2f} bytes/token)")

# ---------- Count all pairs once ----------
pair_counts = dict(count_pairs(tokens, np.arange(len(tokens) - 1)))
print(f"✅ Counted {len(pair_counts)} distinct pairs in {time.time() - start_time:.1f}s")

# ---------- Perform BPE merge iterations ----------
# Each merge only updates the counts of pairs next to the merged positions
for i in range(new_tokens):
    best = max(pair_counts, key=pair_counts.get)
    freq = pair_counts[best]
    if freq < 2:
        print("No pair occurs more than once, stopping early")
        break
    a, b = divmod(best, PAIR_BASE)
    new_id = tokenizer.add_merge(a, b)

    old = tokens
    tokens, pos = merge_pair(old, a, b, new_id)

    # Remove the old pairs that touched the merged tokens ...
    for code, c in count_pairs(old, neighbour_starts([pos - 1, pos, pos + 1], len(old))):
        pair_counts[code] -= c
        if pair_counts[code] == 0:
            del pair_counts[code]
    # ... and add the new pairs around each new token
    new_pos = pos - np.arange(len(pos))
    for code, c in count_pairs(tokens, neighbour_starts([new_pos - 1, new_pos], len(tokens))):
        pair_counts[code] = pair_counts.get(code, 0) + c

    readable = tokenizer.readable_repr(tokenizer.id_to_bytes[new_id])
    print(f"Merge {i + 1}/{new_tokens}: ({a}, {b}) -> {new_id} [{readable}] count={freq}")

print(f"✅ {len(tokens)} tokens after merges ({len(data) / len(tokens):.2f} bytes/token), "
      f"{time.time() - start_time:.1f}s")

# ---------- Save new vocab ----------
tokenizer.save(VOCAB_OUT)
print(f"✅ Saved updated vocab to {VOCAB_OUT}")
