import numpy as np
from net.linear import Linear


x = np.random.randint(-10, 10, (100,30))
model = Linear(n_in=30, n_out=10)
y = model(x)
assert y.shape == (100, 10)

