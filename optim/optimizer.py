import numpy as np

class Optimizer(object):
    '''所有优化器的基类
    '''
    def __init__(self, model, lr=0.001, clip=-1):
        self.params = params_grads
        self.lr = lr
        self.clip = clip

    def zero_grad(self):
        for grad in self.grads:
            grad



