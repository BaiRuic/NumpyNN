from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)
        self.momentum = momentum

    def step(self):
        for params, grads in zip(self.model.params, self.model.grads):
            # print("p:", params)
            # print("g:", grads)
            params -= self.lr * grads
