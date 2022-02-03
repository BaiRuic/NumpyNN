from net.linear import Linear
from net.activation import Sigmoid
from net.sequential import Sequential
from net.loss import MSE
from optim.SGD import SGD
import numpy as np
import matplotlib.pyplot as plt

inputs = np.arange(100).reshape(100,1)
targets = np.sin(inputs)

mod = Sequential(
    Linear(1, 20),
    Sigmoid(),
    Linear(20, 1),
)

criterion = MSE()
optim = SGD(model=mod)

loss_arr = []
for i in range(1000):
    optim.zero_grad()
    outputs = mod(inputs)
    loss = criterion(outputs, targets)
    loss_grad = criterion.backward()
    mod.backward(loss_grad)
    optim.step()
    loss_arr.append(loss)
    if i % 100 == 0:
        print(loss)

plt.plot(loss_arr)
plt.show()



