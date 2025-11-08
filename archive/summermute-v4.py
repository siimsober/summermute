import torch
from s_model import myNN

model = myNN()

# Define loss function (Mean Squared Error) and optimizer (Stochastic Gradient Descent)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Generate training data: y = x + 1
xs = torch.linspace(-10, 10, 100).unsqueeze(1)  # shape [100, 1]
ys = xs + 1

model.load_state_dict(torch.load("models/summermute.pt"))
model.eval()

while True:
    user_input = input("Num: ")
    
    if user_input.lower() in ["quit", "exit"]:
        break
    # Start with the last character of the input
    x = torch.tensor([[ord(user_input[-1])]], dtype=torch.long)
    generated = []

    with torch.no_grad():
        for _ in range(200):  # generate 200 new characters
            logits = model(x)               # [1, 1, 256]
            pred = torch.argmax(logits[0, 0]).item()  # pick most likely next byte
            generated.append(chr(pred))
            x = torch.tensor([[pred]], dtype=torch.long)

    print("Generated:", "".join(generated))