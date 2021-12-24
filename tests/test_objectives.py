from net.loss import MSE
import numpy as np

np.random.seed(0)


preds = np.random.randint(0, 8, (10, 3))
targets = np.random.randint(0, 8, (10, 3))

criterion = MSE()
loss = criterion(preds, targets)
print(loss)