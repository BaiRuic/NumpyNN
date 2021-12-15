import numpy as np
from net.sequential import Sequential
from net.linear import  Linear

fc1 = Linear(5, 10)
fc2 = Linear(10, 20)
fc3 = Linear(20, 2)

x = np.random.rand(100, 5)
assert x.shape == (100, 5), "输入tensor.shape有误"

model = Sequential(fc1, fc2)
y = model(x)
assert y.shape == (100, 20), "输出有误"

model.add(fc3)
y = model(x)
assert y.shape == (100, 2), "加入fc3之后 输出有误"


print(len(model.params))
print(len(model.grads))
print(len(model.params_grads))
