import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#creates a template for the FNN
class FNN(nn.Module):
    def __init__(self) -> None: # runs when you create an instance
        super().__init__() # setup nn.Module as well

        # this creates the feedforward neural network structure
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            #nn.Sigmoid(),
            #nn.ReLU(), 
            nn.Linear(64, 1), 
        )
    
    def forward(self, input):
        return self.net(input)

#create the FNN from the template
model = FNN()

#(3) CREATE THE "TRUE VALUES"/ KUNG SAN MATUTUTO UNG NEURAL NETWORK

#create the training dataset
# MATHEMATICAL FUNCTION: f(x) = 1x^2 + 0x + 0, SCALAR f:R -> R
x_np = np.linspace(start=1, stop=10, num=1000) #set x axis

#y_np = np.polyval(p=[1, 0], x=x_np)
#y_np = np.polyval(p=[1, 0, 0], x=x_np) 
#y_np = np.polyval(p=[1, 0, 0, 0], x=x_np) 

#y_np = 1 / np.polyval(p=[1, 0], x=x_np)

#y_np = np.exp(x_np)
#y_np = np.sqrt(x_np)
#y_np = np.sin(x_np)

#y_np = np.sin(x_np)
#y_np = np.sinh(x_np)
y_np = np.log()

#condlist = [ x_np < 5, x_np >= 5 ]
#funclist = [ lambda x: np.polyval(p=[1, 0, 0], x=x), lambda x: np.polyval(p=[1, 0], x=x)]
#funclist = [ lambda x: np.polyval(p=[1, 0], x=x), lambda x: np.polyval(p=[1, 0, 0], x=x)]
#y_np = np.piecewise(x_np, condlist, funclist)

#convert Numpy Arrays to PyTorch Tensor object
x = torch.tensor(data=x_np, dtype=torch.float32) #shape: (100, )
y = torch.tensor(data=y_np, dtype=torch.float32) #shape: (100, )

#reshape to 2D to be able to do matrix multiplication
x_dataset = x.unsqueeze(1) #shape: (100, 1)
y_dataset = y.unsqueeze(1) #shape: (100, 1)


#print(x.shape)
#print(x)
#print(y.shape)
#print(y)
fig, ax = plt.subplots()
ax.set_xlim(min(x_np), max(x_np))
ax.set_ylim(min(y_np), max(y_np))
ax.set_title("Sigmoid; Sinh(x)")
ax.plot(x_np, y_np, label="True Function", color="red")
#predicted line
line, = ax.plot([], [], label="FNN Copy", color="blue")
#show labels
ax.legend()
predictions = []
losses = []
r2s = []
snapshot_interval = 10


#R^2 

def RR(y_true, y_pred):
    SS_res = torch.sum((y_true - y_pred) ** 2)
    SS_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - SS_res/SS_tot

#(4) TRAIN THE MODEL

#define training tools
loss_function = nn.MSELoss() #Loss function is Mean Squared Error
optimizer = optim.Adam(params=model.parameters(), lr=0.01) #learning rate adjuster is ADAM

#create training loop ---

epochs = 0

while True:
    epochs += 1

    #start training mode
    model.train()

    #clear old gradients
    optimizer.zero_grad()

    #forward pass
    FNN_output = model(x_dataset) #contains matrix multiplication (100, 1) @ (1, 1) @ (1, 1) @ (1, 1) -> (1, 1)

    r2 = RR(y_dataset, FNN_output.detach())
    if r2 >= 0.9999:
        break
    
    #calculate loss
    loss = loss_function(FNN_output, y_dataset)

    #backward pass (compute gradients)
    loss.backward()

    #update weights and bias
    optimizer.step()

  
    if epochs % snapshot_interval == 0:
        pred = FNN_output.squeeze().detach().numpy()
        predictions.append(pred)
        losses.append(loss.item())
        r2s.append(r2)


    if (epochs + 1) % 10 == 0: #only print every 100 loops
        print(f"EPOCH:{epochs}, LOSS:{loss.item():.6f}, R^2:{r2 * 100:.6f}%")

print(f"FINAL NUMBER OF EPOCHS: {epochs}")


epoch_text = ax.text(0.02, 0.05, '', transform=ax.transAxes)
loss_text = ax.text(0.02, 0.10, '', transform=ax.transAxes)
r2_text = ax.text(0.02, 0.15, '', transform=ax.transAxes)


def animate(i):
    line.set_data(x_np, predictions[i])
    epoch_text.set_text(f"EPOCH: { (i+1) * snapshot_interval }")
    loss_text.set_text(f"MSE LOSS: {losses[i]:.6f}")
    r2_text.set_text(f"R^2: {r2s[i]:.6f}")

    #if i == len(predictions) - 1:
    #    anim.event_source.stop()
    #    plt.close()

    return line, epoch_text, loss_text, r2_text

anim = FuncAnimation(fig, animate, frames=range(0, len(predictions), 10), interval=50, blit=True)
#anim = FuncAnimation(fig, animate, frames=len(predictions), interval=50, blit=True)
anim.save("demo.mp4", writer="ffmpeg", fps=10)

#plt.show()


model.eval() #return to evaluate mode (inference mode)

with torch.no_grad(): #disable gradient function to save memory & power
    number = 9 #test number na ipapasok sa neural network
    test_input = torch.tensor([number], dtype=torch.float32).unsqueeze(1) # create test input // should be a 1 x 1 vector
    test_output = model(test_input) # create test output from FNN 
    print(f"TESTING MODEL: INPUT IS {number}, OUTPUT IS: {test_output.item():.6f}")