from .module import Module


class Sequential(Module):
    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = []

        for layer in layers:
            self.layers.append(layer)

    def add(self, other):
        self.layers.append(other)

    def forward(self, input_, *args, **kwargs):
        for layer in self.layers:
            input_ = layer(input_)

        return input_

    def backward(self, pre_grad, *args, **kwargs):
        for layer in self.layers:
            pre_grad = layer.backward(pre_grad)

    @property
    def params(self):
        params = []
        for layer in self.layers:
            params += layer.params

        return params

    @property
    def grads(self):
        grads = []
        for layer in self.layers:
            grads += layer.grads
        return grads
