import numpy as np

class Optimizer(object):
    '''优化器抽象基类
    '''
    def __init__(self, params, lr=0.001, clip=-1):
        self.params = params
        self.lr = lr
        self.clip = clip

    def zero_grad(self):
        for grad in self.grads:
            grad





