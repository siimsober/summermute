import torch
from model import myNN

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
    response = model(torch.tensor([[float(ord(user_input[-1]))]]))
    print("vastus:", chr(int(response.item())))

