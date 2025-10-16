# summermute.py
import sys
import os
import torch
import torch.nn.functional as F
from model import myNN

# ---------- Command-line input ----------
if len(sys.argv) < 2:
    print("Usage: python summermute.py <est or eng> <temperature>")
    sys.exit(1)

model_file = f"models/summermute-{sys.argv[1]}.pt"
if not os.path.exists(model_file):
    print(f"Model file not found: {model_file}")
    sys.exit(1)

model = myNN()
context_len = model.context_len

model.load_state_dict(torch.load(model_file))
model.eval()
print(f"Loaded model: {model_file}")

temperature = float(sys.argv[2])  # >1 = more random, <1 = more confident
max_gen = 200      # max generated characters


while True:
    user_input = input("Sisesta: ")

    
    if user_input.lower() in ["quit", "exit"]:
        break

    # Need at least context length characters of context
    if len(user_input) < context_len:
        user_input = (context_len * " ") + user_input
    
    # Take the last characters as context
    context_bytes = user_input.encode("utf-8")[-context_len:]
    context = list(context_bytes)
    x = torch.tensor([context], dtype=torch.long)

    generated = bytearray()

    with torch.no_grad():
        for _ in range(max_gen):  # generate 200 new characters
            logits = model(x)               # [1, 1, 256]
            logits = logits / temperature

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            pred = torch.multinomial(probs, num_samples=1).item()
            generated.append(pred)
            # Shift context left and append new prediction
            prev = x[0].tolist()[1:]  # drop first byte
            prev.append(pred)         # add new predicted byte
            x = torch.tensor([prev], dtype=torch.long)

    print("Generated:", generated.decode("utf-8", errors="replace"))