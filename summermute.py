import torch

class myNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        #with torch.no_grad():
        #    self.linear.weight.fill_(1.0)
        #    self.linear.bias.fill_(1.0)

    def forward(self, x):
        return self.linear(x)

model = myNN()

# Define loss function (Mean Squared Error) and optimizer (Stochastic Gradient Descent)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Generate training data: y = x + 1
xs = torch.linspace(-10, 10, 100).unsqueeze(1)  # shape [100, 1]
ys = xs + 1

# Train for 500 epochs
for epoch in range(5000):
    print("Treenimine epohh 1")
    # Forward pass
    outputs = model(xs)
    loss = criterion(outputs, ys)
    print("kadu", loss)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

while True:
    user_input = input("Num: ")
    
    if user_input.lower() in ["quit", "exit"]:
        break
    response = model(torch.tensor([[float(ord(user_input[-1]))]]))
    print("sinu number + 1:", chr(int(response.item())))

