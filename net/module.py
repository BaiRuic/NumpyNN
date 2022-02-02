class Module(object):
    """
    为所有网络的基类
    """
    def __init__(self):
        pass

    def __call__(self, input_):
        return self.forward(input_)

    def forward(self, input_):
        raise NotImplementedError

    def backward(self, pre_grad):
        '''
        计算当前网络参数的梯度信息，并向前一层返回
        Parameters
        ----------
        pre_grad
        上一层网络传过来的梯度信息
        注：此出的上一层是按照反向传播的顺便
        Returns
        -------
        梯度
        '''
        raise NotImplementedError

    @property
    def params(self):
        '''
        返回当前网络的参数
        Returns
        -------
        当前网络的参数
        '''
        return []

    @property
    def grads(self):
        '''
        返回当前网络参数的梯度
        Returns
        -------
        参数的梯度
        '''
        return []

    @property
    def params_grads(self):
        '''
        返回当前网络的参数以及对应的梯度
        Returns
        -------
        当前网络对应的参数及梯度
        '''
        return list(zip(self.params, self.grads))

