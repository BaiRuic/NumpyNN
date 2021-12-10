import numpy as np

class Sigmoid(object):
    def __init__(self):
        self.output = None

    def forward(self, input, *args, **kwargs):
        self.output = 1.0 / (1 + np.exp(-input))
        return self.output

    def backward(self, preGrad):
        return np.multiarray(preGrad, np.multiarray(self.output, 1.0-self.output))


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
        return output

    def backward(self, preGrad=None):
        preGrad[self.mask] = 0
        return preGrad

class SoftMax(object):
    def __init__(self):

    def forward(self, input):
        '''计算输出的softmax，防止溢出，在计算前，减去了每一个样本的最大输出值
        Parameters
        ----------
        input:[np.array]

        Returns np.array
        -------
        '''
        max_val = np.max(input, axis=0)
        exp = np.exp(input-max_val)
        sum_exp = np.sum(exp, axis=0)
        res = exp / sum_exp
        return res

    def backward(self, preGrad=1):


class crossEntropyError(object):
    '''当出现np.log(0)时，np.log(0)会变为负无限大,所以做为防护性措施，提前加一个较小值

    '''
    def __init__(self, delta=1e-7):
        self.delta = delta

    def forward(self, pred, target):
        '''当求单个数据的交叉熵误差时，需要改变数据的形状。
            并且，当输入为mini-batch 时，要用batch 的个数进行正规化，计算单个数据的平均交叉熵误差。

        Parameters
        ----------
        pred:神经网络的输出
        target：标签

        Returns 交叉熵损失值
        -------

        '''
        if pred.ndim == 1:
            pred.reshape(1, pred.size)
            target.reshape(1, target.size)
        batch_size = target.shape[0]

        # 如果 target是one-hot编码形式, 直接乘
        if target.shape[1] == pred.shape[1]:
            res = - np.sum(target * np.log(pred + self.delta)) / batch_size

        # 不是 one-hot 编码形式，先正确解标签处的输出
        elif target.shape[1] == 1:
            pred = [np.range(batch_size), target]
            res = - np.sum(np.log(pred + self.delta)) / batch_size

    def backward(self, preGrad=1):





