import numpy as np
from .module import Module


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
        pass

    def forward(self, input_):
        '''计算输出的softmax，防止溢出，在计算前，减去了每一个样本的最大输出值
        Parameters
        ----------
        input_:[np.array]

        Returns np.array
        -------
        '''
        max_val = np.max(input_, axis=1)
        exp = np.exp(input_ - max_val)
        sum_exp = np.sum(exp, axis=1)
        res = exp / sum_exp
        return res

    def backward(self, preGrad=1):
        pass


class crossEntropyError(Module):
    '''当出现np.log(0)时，np.log(0)会变为负无限大,所以做为防护性措施，提前加一个较小值

    '''
    def __init__(self, delta=1e-7):
        super(crossEntropyError, self).__init__()
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

        return res

    def backward(self, pre_grad=1):
        pass






