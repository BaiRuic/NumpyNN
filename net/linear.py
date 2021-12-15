import numpy as np
from numpy.lib.type_check import _nan_to_num_dispatcher

class Linear(object):
    '''Full connected layer

    Parameters
    -----------
    n_out:
        the size of the current layer output
    n_in:
        the size of current layer needed feed
    '''
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out

        self.W = None
        self.b = None

        self.dW = None
        self.db = None
        self.curInput = None

        self.firstLayer = False

        self._initParams(self.n_in, self.n_out)

    
    def _initParams(self, n_in, n_out):
        '''初始化参数为 正态分布，均值0 方差 np.sqrt(2 / (fan_out + fan_in))
        '''
        mean = 0.0
        std = np.sqrt(2 / (n_in + n_out))
        self.W = np.random.normal(mean, std, (n_in, n_out))
        self.b = np.zeros((n_out))

    def connectPrevLayer(self, prevLayer=None):
        if prevLayer is None:
            n_in = self.n_in
        else:
            n_in = prevLayer.outShape[-1]
        
        n_out = self.n_out
        self._initParams(n_in, n_out)

    def forward(self, input, *args, **kwargs):
        self.curInput = input
        return np.dot(input, self.W) + self.b

    def backward(self, preGrad, *args, **kwargs):
        self.dW = np.dot(self.curInput.T, preGrad)
        self.db = np.mean(preGrad, axis=0)
        if self.firstLayer is False:
            return np.dot(preGrad, self.W.T)

    @property
    def params(self):
        return (self.W, self.b)

    @property
    def grads(self):
        return (self.dW, self.db)

    @property
    def paramsGrads(self):
        return (zip(self.params, self.grads))

class Sequence(object):
    def __init__(self, *layer):

