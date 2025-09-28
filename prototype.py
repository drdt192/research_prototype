import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def RR(y_true, y_pred):
    SS_res = torch.sum((y_true - y_pred) ** 2)
    SS_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - SS_res/SS_tot

exit = "N"

while exit == "N":
    available = [
        "Linear",
        "Quadratic",
        "Cubic",
        "Rational",
        "Exponential",
        "Radical",
        "Logarithmic",
        "Trigonometric",
        "Hyperbolic",
    ]

    print(". . .prototype_alpha_v1.1 by drdt192. . .\n\n")

    while True:
        try:
            perceptron_count = int(input(f"How many perceptrons are within the single hidden layer? Choose from 32, 64, and 128!\n"))
        except ValueError:
            print("That is not a number!")
            continue
        if perceptron_count in [32, 64, 128]: break
        print("That isn't a valid perceptron count!")


    while True:
        function = input(f"Hello! Which scalar mathematical function would you like the FNN to learn?\n {available}\n")
        if function in available: break
        print("That is not a valid function! Please try again!\n")

    while True:
        try:
            trials = int(input("How many trials would you like to do? Choose from 1 to 100!\n"))
        except ValueError:
            print("That is not a number!\n") 
            continue
        if trials in list(range(1, 101)): break
        print("That is not a valid number of trials! Please try again!\n")

    trial = 1
    results = []

    print("Starting... Please wait until 99.99% for each trial!")
    while trial <= trials:
        class FNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.net = nn.Sequential(
                    nn.Linear(1, perceptron_count),
                    nn.ReLU(), 
                    nn.Linear(perceptron_count, 1), 
                )
    
            def forward(self, input):
                return self.net(input)
        model = FNN()

        x_np = np.linspace(start=1, stop=10, num=1000)
        y_np = {
            "Linear": np.polyval(p=[1, 0], x=x_np),
            "Quadratic": np.polyval(p=[1, 0, 0], x=x_np),
            "Cubic": np.polyval(p=[1, 0, 0, 0], x=x_np),
            "Rational":  1 / np.polyval(p=[1, 0], x=x_np),
            "Exponential": np.exp(x_np),
            "Radical": np.sqrt(x_np),
            "Logarithmic": np.log(x_np),
            "Trigonometric": np.sin(x_np),
            "Hyperbolic": np.sinh(x_np),
        }

        x = torch.tensor(data=x_np, dtype=torch.float32)
        y = torch.tensor(data=y_np[function], dtype=torch.float32)
        x_dataset = x.unsqueeze(1)
        y_dataset = y.unsqueeze(1)

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(params=model.parameters(), lr=0.1)

        epochs = 0
        while epochs < 1_000_000:
            epochs += 1
            model.train()
            optimizer.zero_grad()
            FNN_output = model(x_dataset)

            r2 = RR(y_dataset, FNN_output.detach())
            if r2 >= 0.9999:
                break
    
            loss = loss_function(FNN_output, y_dataset)
            loss.backward()
            optimizer.step()

            if (epochs + 1) % 10 == 0:
                print(f"EPOCH:{epochs}, LOSS:{loss.item():.6f}, R^2:{r2 * 100:.6f}%")

        print(f">FUNCTION TYPE: {function}")
        print(f">FINAL NUMBER OF EPOCHS: {epochs}, TRIAL#{trial}\n\n")
        results.append(epochs)
        trial += 1

    mean = np.mean(results)
    residuals = [(result - mean)**2 for result in results]
    variance = np.mean(residuals)
    std_var = np.sqrt(variance)
    print(f"Here are the results! Please copy.\n{results}\n")
    print(f"FUNCTION: {function}, PERCEPTRONS: {perceptron_count}, TRIALS: {trials}\n")
    print(f"MEAN: {mean}, VAR: {variance}, STD_VAR: {std_var}\n")

    while True:
        prompt = input("Would you like to use the prototype again? Please type 'Y' or 'N'.\n")
        if prompt in ["Y", "N"]: 
            exit = prompt
            break
        print("That is not a valid input!\n")

input("Thank you for using the prototype! Press ENTER to exit!")
input("Are you sure you want to exit? Press ENTER again!")