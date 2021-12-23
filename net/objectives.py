from .module import Module

class crossEntropyError(Module):
    '''当出现np.log(0)时，np.log(0)会变为负无限大,所以做为防护性措施，提前加一个较小值

    '''
    def __init__(self, delta=1e-7):
        super(crossEntropyError, self).__init__()
        self.pred = None
        self.target = None
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
        # 如果 target是one-hot编码形式, 直接乘
        if target.shape[1] == pred.shape[1]:
            res = - np.sum(target * np.log(pred + self.delta)) / batch_size

        # 不是 one-hot 编码形式，先正确解标签处的输出
        elif target.shape[1] == 1:
            pred = [np.range(batch_size), target]
            res = - np.sum(np.log(pred + self.delta)) / batch_size

        return res

    def backward(self, pre_grad=1):
        return