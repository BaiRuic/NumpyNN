import numpy as np
from npnn.linear import Linear


class PrevLayer:
    def __init__(self, outShape):
        self.outShape = outShape


# 测试 Linear 层
l = Linear(4, 8)
l.connectPrevLayer(None)

assert(l.W.shape==(4, 8))
assert(l.b.shape == (8,))

l.connectPrevLayer(PrevLayer((20, 4)))
assert(l.W.shape==(4, 8))
assert(l.b.shape == (8,))

input = np.random.rand(100, 4)
assert(l.forward(input).shape==(100,8))


pre_grad = np.random.rand(100, 8)
assert(l.backward(preGrad=pre_grad).shape==input.shape)



