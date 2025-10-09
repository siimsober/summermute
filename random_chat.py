import random

responses = [
    "Hello, human!",
    "I am Summermute, your AI.",
    "I don't quite understand that.",
    "Interesting… tell me more."
]

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    print("Summermute:", random.choice(responses))