import numpy as np
from .module import Module


class Linear(Module):
    """Full connected layer

    Parameters
    -----------
    n_out:
        the size of the current layer output
    n_in:
        the size of current layer needed feed
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        self.n_in = n_in
        self.n_out = n_out

        self.W = None
        self.b = None

        self.dW = None
        self.db = None

        self._input = None
        self._output = None

        self._init_params(self.n_in, self.n_out)

    def _init_params(self, n_in, n_out):
        """初始化参数为 正态分布，均值0 方差 np.sqrt(2 / (fan_out + fan_in))
        """
        mean = 0.0
        std = np.sqrt(2 / (n_in + n_out))
        self.W = np.random.normal(mean, std, (n_in, n_out))
        self.b = np.zeros(n_out)

    def forward(self, input_):
        self._input = input_
        self._output = np.dot(input_, self.W) + self.b
        return self._output

    def backward(self, pre_grad):
        self.dW = np.dot(self._input.T, pre_grad)
        self.db = np.mean(pre_grad, axis=0)

        return np.dot(pre_grad, self.W.T)

    @property
    def params(self):
        return [self.W, self.b]

    @property
    def grads(self):
        return [self.dW, self.db]
