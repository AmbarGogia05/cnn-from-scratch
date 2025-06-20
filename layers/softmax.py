import numpy as np

# For a classification problem, softmax converts the values into probabilities
class SoftMax:
    def __init__(self):
        pass

    def forward(self, x):
        x = x - np.max(x) # Doesn't change the result, just results in smaller exponentials preventing potential overflow issues
        exp_x = np.exp(x)
        probabilities = exp_x / np.sum(exp_x)
        return probabilities
    
    def backward(self, dout):
        return dout
    
    # This assumes softmax is used in conjunction with cross entropy always
    