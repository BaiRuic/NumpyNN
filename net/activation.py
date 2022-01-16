import numpy as np
from net.module import Module


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.output = None

    def forward(self, input_):
        self.output = 1.0 / (1 + np.exp(-input_))
        return self.output

    def backward(self, pre_grad):
        return np.multiply(pre_grad, np.multiply(self.output, 1.0-self.output))


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.output = None

    def forward(self, input_):
        self.output = np.tanh(input_)
        return self.output

    def backward(self, pre_grad):
        return np.multiply(pre_grad, 1.0 - np.power(self.output, 2))


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.mask = None

    def forward(self, input_):
        self.mask = input_ <= 0
        output = input_.copy()
        output[self.mask] = 0
        return output

    def backward(self, pre_grad=None):
        # temp = np.ones_like(pre_grad)
        # temp[self.mask] = 0
        # return np.multiply(pre_grad, temp)
        pre_grad[self.mask] = 0
        return pre_grad


class SoftMax(Module):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.res = None

    def forward(self, input):
        '''计算输出的softmax，防止溢出，在计算前，减去了每一个样本的最大输出值
        Parameters
        ----------
        input:[np.array]

        Returns np.array
        -------
        '''
        max_val = np.max(input, axis=1)
        exp = np.exp(input - max_val)
        sum_exp = np.sum(exp, axis=1)
        self.res = exp / sum_exp
        return self.res

    def backward(self, pre_grad=1):
        dout = np.diag(self.res) - np.outer(self.res, self.res)
        return np.dot(dout, pre_grad)