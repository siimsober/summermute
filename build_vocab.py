import sys
import collections
import os
import re
from tokenizer import BPETokenizer

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


if len(sys.argv) < 3:
    print("Usage: python make_vocab.py <est|eng> <number of tokens to add>")
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

# ---------- Perform BPE merge iterations ----------
for i in range(new_tokens):
    print(f"\n--- Merge iteration {i + 1}/{new_tokens} ---")

    token_ids = tokenizer.encode(data)

    pair_counts = collections.Counter(
        (token_ids[i], token_ids[i + 1]) for i in range(len(token_ids) - 1)
    )

    most_common_pair, freq = pair_counts.most_common(1)[0]
    print(f"Most frequent pair: {most_common_pair} (count={freq})")

    # Merge bytes for this pair
    pair_bytes = tokenizer.id_to_bytes[most_common_pair[0]] + tokenizer.id_to_bytes[most_common_pair[1]]
    print(f"Bytes: {pair_bytes}")
    #print("build: ", repr(pair_bytes), type(pair_bytes))
    print(f"Readable: [{tokenizer.readable_repr(pair_bytes)}]")

    tokenizer.add_token(pair_bytes)

# ---------- Save new vocab ----------
tokenizer.save(VOCAB_OUT)
print(f"✅ Saved updated vocab to {VOCAB_OUT}")