import sys
import os
import torch
from model import myNN

context_len = 2

# ---------- Command-line input ----------
if len(sys.argv) < 2:
    print("Usage: python train.py <est or eng>")
    sys.exit(1)

model_file = f"models/summermute-{sys.argv[1]}.pt"
if not os.path.exists(model_file):
    print(f"Model file not found: {model_file}")
    sys.exit(1)

model = myNN()

model.load_state_dict(torch.load(model_file))
model.eval()
print(f"Loaded model: {model_file}")

while True:
    user_input = input("Sisesta: ")

    
    if user_input.lower() in ["quit", "exit"]:
        break

    # Need at least 2 characters of context
    if len(user_input) < context_len:
        user_input = (context_len * " ") + user_input
    
    # Take the last two characters as context
    context = [ord(c) for c in user_input[-2:]]
    x = torch.tensor([context], dtype=torch.long)  # shape [1, 2]

    generated = bytearray()

    with torch.no_grad():
        for _ in range(200):  # generate 200 new characters
            logits = model(x)               # [1, 1, 256]
            pred = torch.argmax(logits, dim=-1).item()
            generated.append(pred)
            x = torch.tensor([[x[0, 1].item(), pred]], dtype=torch.long)

    print("Generated:", generated.decode("utf-8", errors="replace"))