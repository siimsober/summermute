# sumerute.py
import sys
import os
import torch
import torch.nn.functional as F
from s_model import myNN
from tokenizer import BPETokenizer, EOT_ID
import re
import os

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
if len(sys.argv) < 3:
    print("Usage: python sumerute.py <est or eng> <temperature>")
    sys.exit(1)

lang = sys.argv[1]
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

temperature = float(sys.argv[2])  # >1 = more random, <1 = more confident
print(f"Running at temp: {temperature}")
max_gen = 200      # max generated tokens


while True:
    user_input = input("Sisesta: ")

    
    if user_input.lower() in ["quit", "exit"]:
        break

    # Need at least context length characters of context
    if len(user_input) < 1:
        user_input = " "
    
    # Take the last characters as context
    context_bytes = user_input.encode("utf-8")
    #print(context_bytes)
    context = tokenizer.encode(context_bytes)
    #print(context)
    # pad context to context_len tokens
    #if len(context) < context_len:
    #    context = [32]*(context_len - len(context)) + context  # pad with spaces on left
    #else:
    #    context = context[-context_len:]  # truncate to last context_len tokens
    #print(context)
    context = torch.tensor(context, dtype=torch.long)[None, :]  # initial input [1, T]
    generated = []

    with torch.no_grad():
        for _ in range(max_gen):
            # keep only the last max_len tokens
            x_input = context[:, -model.max_len:]
            logits = model(x_input)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == EOT_ID:
                break  # model ended the text

            # append to context for next iteration
            context = torch.cat([context, next_token], dim=1)
            generated.append(next_token.item())
            
    generated_bytes = tokenizer.decode(generated)
    print("Generated:", generated_bytes.decode("utf-8", errors="replace"))