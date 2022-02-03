from .optimizer import Optimizer

class SGD(Optimizer):
    """
    随机梯度下降
    """
    def __init__(self, momentum=0.9, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)
        self.momentum = momentum

    def step(self):
        for params, grads in zip(self.model.params, self.model.grads):
            params -= self.lr * grads
