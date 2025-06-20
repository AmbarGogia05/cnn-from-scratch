import numpy as np

class ReLU:
    def __init__(self):
        self.x = None
        
    def forward(self, x):
        self.x = x
    # Here, x is a numpy array of arbitrary dimension
        return np.maximum(0, x)
    def backward(self, dout):
        return dout * (self.x>0)