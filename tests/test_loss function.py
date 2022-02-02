import numpy as np

from net.loss import MSE

outputs = np.random.randint(2, 4, size=(1, 2))
target = np.random.randint(2, 4, size=(1, 2))

print(outputs, "\n", target)
criterion = MSE()

loss = criterion(outputs, target)
print(loss)
pre_grads = criterion.backward()
print(pre_grads.shape)

