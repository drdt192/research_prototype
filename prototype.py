#for file management
import csv
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog

#core libraries used
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("<<< prototype_alpha_v2.0 by drdt192 >>>\n\n")

#R-squared accuracy 
def RR(y_true, y_pred):
    SS_res = torch.sum((y_true - y_pred) ** 2)
    SS_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - SS_res/SS_tot

#Loss function that determines the error of the FNN
loss_function = nn.MSELoss()

#all x-values dataset
x_np = np.linspace(start=1, stop=10, num=1000)

#available scalar mathematical functions (for the FNN to mimic)
scalar_types = {
    "Linear": lambda x_np: np.polyval(p=[1, 0], x=x_np),
    "Quadratic": lambda x_np: np.polyval(p=[1, 0, 0], x=x_np),
    "Cubic": lambda x_np: np.polyval(p=[1, 0, 0, 0], x=x_np),
    "Rational": lambda x_np: 1 / np.polyval(p=[1, 0], x=x_np),
    "Exponential": lambda x_np: np.exp(x_np),
    "Radical": lambda x_np: np.sqrt(x_np),
    "Logarithmic": lambda x_np: np.log(x_np),
    "Trigonometric": lambda x_np: np.sin(x_np),
    "Hyperbolic": lambda x_np: np.sinh(x_np)
}

#available non-linear activation functions (for the FNN internals)
activation_types = {
    "Sigmoid": nn.Sigmoid,
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh
}

#initiates tkinter window (for saving the file)
root = tk.Tk()
root.withdraw()

#main experiment loop
exit = "N"
while exit == "N":
    
    while True:
        print(f">>>CHOOSE YOUR SCALAR FUNCTION TYPE:\n{list(scalar_types.keys())}\n\n")
        scalar = simpledialog.askstring("Input", "SCALAR?")
        if scalar in list(scalar_types.keys()): break
        print("That is not a valid function! Please try again!\n")

    while True:
        print(f'>>>CHOOSE YOUR ACTIVATION FUNCTION TYPE:\n{list(activation_types.keys())}\n\n')
        activation = simpledialog.askstring("Input", "ACTIVATION?")
        if activation in list(activation_types.keys()): break
        print("That is not a valid activation! Please try again!\n")

    while True:
        #try:

        print(">>>CHOOSE YOUR TRIAL COUNT: 1 TO 100\n\n")
        trials = simpledialog.askinteger("Input", "TRIALS?")

        #except ValueError:
        #    print("That is not a number! Please try again!\n") 
        #    continue

        if trials in list(range(1, 101)): break
        print("That is not a valid trial count! Please try again!\n")

    #initialize list to be packed into .csv
    results = [["SCALAR", "ACTIVATION", "EPOCH"]]

    print("Starting... Please wait until 99.99% R^2 for each trial!")
    #n-trial loop
    trial = 0
    while trial < trials:
        try:
            #Creates a new FNN for each trial, new parameters each time
            class FNN(nn.Module):
                def __init__(self) -> None:
                    super().__init__()

                    self.net = nn.Sequential(
                        #The single hidden layer has 64 perceptrons
                        nn.Linear(1, 64),
                        activation_types[activation](), 
                        nn.Linear(64, 1), 
                    )
    
                def forward(self, input):
                    return self.net(input)
            model = FNN()

            #Get the y-values dataset by inputting the x-values into the scalar function chosen
            y_np = scalar_types[scalar](x_np)

            #Convert into torch tensor to be able to use the library
            x = torch.tensor(data=x_np, dtype=torch.float32)
            y = torch.tensor(data=y_np, dtype=torch.float32)
            #Expand into 2D vector for matmul
            x_dataset = x.unsqueeze(1)
            y_dataset = y.unsqueeze(1)
            
            #Slowly shifts the learning rate 
            optimizer = optim.Adam(params=model.parameters(), lr=0.05)

            epochs = 0
            #initiate main training loop
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

                if epochs % 100 == 0:
                    print(f"EPOCH:{epochs}, LOSS:{loss.item():.6f}, R^2:{r2 * 100:.6f}% - CTRL+C TO EXIT!")
            trial+=1


            #display results after each successful loop to 99.99% R^2
            print(f">SCALAR TYPE: {scalar}, >EPOCHS: {epochs}, TRIAL#{trial}\n\n")

            #append to .csv list
            #[scalar_type, activation_type, epochs]
            results.append([list(scalar_types).index(scalar)+1, list(activation_types).index(activation)+1, epochs])
 
        #when user does Ctrl+C, gets an option to terminate training
        except KeyboardInterrupt:
            while True:
                print(">>>Would you to TERMINATE the experiment? Please type 'Y' or 'N'.\n")
                prompt = simpledialog.askstring("Input", "TERMINATE?")
                if prompt in ["Y", "N"]: 
                    exit = prompt
                    break
                print("That is not a valid input!\n")
            if prompt == "Y":
                print(f"INTERRUPTED AT TRIAL#{trial}\n")
                break

    #display all results of all trials     
    if results:
        #ask where to save the file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save data as?",
            initialfile=f"{scalar}_{activation}_{trial}.csv"
        )

        if file_path:
            print("Save path:", file_path)
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(results)
        else:
            print("Save cancelled.")

        print(f"Here are the results!\n{results}\n")
        print(f"SCALAR: {scalar}, ACTIVATION: {activation}, TRIALS COMPLETED: {len(results)-1}/{trials}\n")

    #given the option to try again or exit
    while True:
        print(">>>Would you like to close the program? Please type 'Y' or 'N'.\n(Choosing 'N' would start another experiment)\n\n")
        prompt = simpledialog.askstring("Input", "CLOSE?")
        if prompt in ["Y", "N"]: 
            exit = prompt
            break
        print("That is not a valid input!\n")

root.destroy()