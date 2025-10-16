import sys
import collections
from tokenizer import BPETokenizer
import os
import re

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
INPUT_FILE = f"data/training-{lang}.txt"   # training file
latest_vocab = find_latest_vocab_number("data/vocab", lang)
VOCAB_FILE = f"data/vocab/vocab-{lang}-{latest_vocab}.tsv"  # your existing vocab
VOCAB_OUT = f"data/vocab/vocab-{lang}-{latest_vocab + new_tokens}.tsv"

# ---------- Load tokenizer ----------
tokenizer = BPETokenizer(VOCAB_FILE)
print("Tokenizer initialized.")

# ---------- Read file as bytes ----------
with open(INPUT_FILE, "rb") as f:
    data = f.read()  # bytearray of entire file
print("Data read.")

for _ in range(new_tokens):
    # ---------- Encode to token IDs ----------
    token_ids = tokenizer.encode(data)
    print(f"Total tokens: {len(token_ids)}")

    # ---------- Count adjacent token pairs ----------
    pair_counts = collections.Counter()
    for i in range(len(token_ids) - 1):
        pair_counts[(token_ids[i], token_ids[i + 1])] += 1

    # ---------- Find most frequent pair ----------
    most_common_pair, freq = pair_counts.most_common(1)[0]
    print(f"Most frequent token pair: {most_common_pair} (count {freq})")

    # Optional: print human-readable representation
    pair_bytes = tokenizer.id_to_bytes[most_common_pair[0]] + tokenizer.id_to_bytes[most_common_pair[1]]
    print(f"Bytes: {pair_bytes}")
    print(f"Readable: [{tokenizer.readable_repr(pair_bytes)}]")
    tokenizer.add_token(pair_bytes)

tokenizer.save(VOCAB_OUT)