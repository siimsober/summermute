# summermute.py
import sys
import os
import torch
import torch.nn.functional as F
from model import myNN
from tokenizer import BPETokenizer
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
    print("Usage: python summermute.py <est or eng> <temperature>")
    sys.exit(1)

lang = sys.argv[1]
latest_num = find_latest_vocab_number("data/vocab", lang)
vocab_path = f"data/vocab/vocab-{lang}-{latest_num}.tsv"
model_file = f"models/summermute-{lang}.pt"
if not os.path.exists(model_file):
    print(f"Model file not found: {model_file}")
    sys.exit(1)

tokenizer = BPETokenizer(vocab_path)
model = myNN(tokenizer.get_vocab_size())
#context_len = model.context_len
#print(f"Model context length: {context_len}")

model.load_state_dict(torch.load(model_file))
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
    x = torch.tensor([context], dtype=torch.long)
    hidden = None
    generated = []

    with torch.no_grad():
        # First forward pass: process the whole input
        logits, hidden = model(x, hidden)

        # Start generation from the last token
        next_token = x[0, -1].unsqueeze(0).unsqueeze(0)

        for _ in range(max_gen):  # generate 200 new tokens
            logits, hidden = model(next_token, hidden)               
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # shape [1, 1]
            generated.append(next_token.item())
            
    generated_bytes = tokenizer.decode(generated)
    print("Generated:", generated_bytes.decode("utf-8", errors="replace"))