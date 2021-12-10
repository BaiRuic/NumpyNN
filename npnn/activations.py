import numpy as np

class Sigmoid(object):
    def __init__(self):
        self.output = None

    def forward(self, input, *args, **kwargs):
        self.output = 1.0 / (1 + np.exp(-input))
        return self.output

    def backward(self, preGrad):
        return np.multiarray(preGrad, np.multiarray(self.output, 1-self.output))


class Tanh(object):
    def __init__(self):
        self.output = None

    def forward(self, input):
        self.output = np.tanh(input)
        return self.output

    def backward(self, preGrad):
        return np.multiarray(preGrad, 1.0 - np.power(self.output, 2))


class ReLU(object):
    def __init__(self):
        self.mask = None

    def forward(self, input):
        self.mask = (input <= 0)
        output = input.copy()
        output[self.mask] = 0
        return self.mask

    def backward(self, preGrad=None):
        preGrad = preGrad if preGrad else
        preGrad[self.mask] = 0
        res = preGrad
        return res


