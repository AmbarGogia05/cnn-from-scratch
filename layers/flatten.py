import numpy as np

class Flatten:
    def __init__(self):
        self.original_shape = None

    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(-1)

    def backward(self, dout):
        return dout.reshape(self.original_shape)
