# sumerute.py
import sys
import argparse
import os
import torch
import torch.nn.functional as F
from s_model import myNN
from tokenizer import BPETokenizer, EOT_ID
import re

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

# ---------- Command-line input ----------
parser = argparse.ArgumentParser(description="Chat with a trained sumerute model")
parser.add_argument("lang", choices=["est", "eng"])
parser.add_argument("temperature", type=float, help=">1 = more random, <1 = more confident")
parser.add_argument("--top-k", type=int, default=40, help="sample only from the k most likely tokens, 0 = off")
parser.add_argument("--top-p", type=float, default=0.95,
                    help="sample only from the most likely tokens covering this probability, 1 = off")
args = parser.parse_args()

lang = args.lang
model_file = f"models/sumerute-{lang}.pt"
if not os.path.exists(model_file):
    print(f"Model file not found: {model_file}")
    sys.exit(1)

checkpoint = torch.load(model_file, map_location="cpu")
if "config" in checkpoint:
    # Checkpoint from s_train.py: model sizes and the vocab it was trained with
    config = checkpoint["config"]
    state = checkpoint["model"]
    vocab_path = config["vocab_path"]
    print(f"Checkpoint from step {checkpoint['step']}, val loss {checkpoint['best_val']:.4f}")
else:
    # Older files are a bare state_dict with default sizes and the latest vocab
    config = dict(embed_size=64, max_len=256, dropout=0.1)
    state = checkpoint
    vocab_path = f"data/vocab/vocab-{lang}-{find_latest_vocab_number('data/vocab', lang)}.tsv"

tokenizer = BPETokenizer(vocab_path)
model = myNN(tokenizer.get_vocab_size(), config["embed_size"], config["max_len"], config["dropout"])
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}, vocab {vocab_path}")

model.load_state_dict(state)
model.eval()
print(f"Loaded model state: {model_file}")

temperature = args.temperature
print(f"Running at temp {temperature}, top-k {args.top_k or 'off'}, top-p {args.top_p if args.top_p < 1 else 'off'}")
max_gen = 200      # max generated tokens


def sample_next(logits):
    """Pick the next token from the last position's logits with temperature, top-k and top-p."""
    logits = logits / temperature
    if args.top_k > 0:
        # Drop everything below the k-th largest logit
        kth = torch.topk(logits, min(args.top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if args.top_p < 1.0:
        # Keep the smallest set of most likely tokens whose probabilities add up to top_p
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum_probs - F.softmax(sorted_logits, dim=-1) >= args.top_p   # always keeps the top token
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


while True:
    user_input = input("Sisesta: ")

    if user_input.lower() in ["quit", "exit"]:
        break
    if len(user_input) < 1:
        user_input = " "

    context = tokenizer.encode(user_input.encode("utf-8"))
    context = torch.tensor(context, dtype=torch.long)[None, :]  # initial input [1, T]
    generated = []
    stop_reason = f"length limit ({max_gen} tokens)"

    with torch.no_grad():
        for _ in range(max_gen):
            # keep only the last max_len tokens
            x_input = context[:, -model.max_len:]
            logits = model(x_input)
            next_token = sample_next(logits[:, -1, :])
            if next_token.item() == EOT_ID:
                stop_reason = "end of text (EOT)"
                break

            # append to context for next iteration
            context = torch.cat([context, next_token], dim=1)
            generated.append(next_token.item())

    generated_text = tokenizer.decode(generated).decode("utf-8", errors="replace")
    print(f"\n{user_input}{generated_text}")
    print(f"[{len(generated)} tokens, stopped: {stop_reason}]\n")
