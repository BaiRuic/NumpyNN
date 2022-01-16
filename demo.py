from net.linear import Linear
from net.activation import ReLU
from net.sequential import Sequential
from net.loss import MSE
import numpy as np


inputs = np.arange(100).reshape(100,1)
targets = np.sin(inputs)

model = Sequential(
    Linear(1, 8),
    ReLU(),
    Linear(8, 1),
)

criterion = MSE()
optim = SGD()


