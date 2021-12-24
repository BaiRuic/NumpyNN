# -*- coding: utf-8 -*-
"""
神经网络中常用的损失函数
"""

import numpy as np


class Objective(object):
    """损失函数基类
    """
    def __init__(self):
        self.preds = None
        self.targets = None

    def __call__(self, preds, targets):
        return self.forward(preds, targets)

    def forward(self, preds, targets):
        """前向传播，计算损失值

        Parameters
        ----------
        preds: type: [numpy.array] shape:[batch_size, features]
            模型预测结果
        targets:type: [numpy.array] shape:[batch_size, features]
            真实值

        Returns
        -------
        numpy.array
            损失值
        """
        raise NotImplementedError

    def backward(self, pre_grad=1.0):
        """反向传播函数

        Parameters
        ----------
        pre_grad: numpy.array
            由上一层上下来的损失值，一般为1.0

        Returns
        -------
        numpy.array
            梯度
        """
        raise NotImplementedError


class MSE(Objective):
    """

    """
    def forward(self, preds, targets):
        self.preds = preds
        self.target = targets
        # batch_size = preds.shape[0]
        # return 0.5 * np.sum(np.power(preds - targets, 2)) / batch_size
        return 0.5 * np.mean(np.sum(np.power(preds - targets, 2), axis=1))

    def backward(self):
        return self.preds - self.target



class crossEntropyError(Objective):
    '''当出现np.log(0)时，np.log(0)会变为负无限大,所以做为防护性措施，提前加一个较小值

    '''
    def __init__(self, delta=1e-7):
        super(crossEntropyError, self).__init__()
        self.preds = None
        self.targets = None
        self.delta = delta

    def forward(self, preds, targets):
        '''当求单个数据的交叉熵误差时，需要改变数据的形状。
            并且，当输入为mini-batch 时，要用batch 的个数进行正规化，计算单个数据的平均交叉熵误差。

        Parameters
        ----------
        preds:神经网络的输出
        targets：标签

        Returns 交叉熵损失值
        -------

        '''
        res = None
        # 如果 target是one-hot编码形式, 直接乘
        if targets.shape[1] == preds.shape[1]:
            res = - np.sum(targets * np.log(preds + self.delta)) / batch_size


        # 不是 one-hot 编码形式，先正确解标签处的输出
        elif targets.shape[1] == 1:
            batch_size = targets.shape[0]
            preds = [np.range(batch_size), targets]
            res = - np.sum(np.log(preds + self.delta)) / batch_size

        return res

    def backward(self, pre_grad=1):
        pass