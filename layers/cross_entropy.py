import numpy as np
# Here, we calculate the cross entropy loss for the output from the dense layer after applying softmax
class CrossEntropy:
    def __init__(self):
        pass

    def forward(self, x, y):
        self.x = x
        self.y = y
        #x is the array of predicted probabilities, while y consists of true labels
        cross_entropy = -np.sum(y*(np.log(x+1e-15))) # A small constant is added to prevent instability when probability is very close to 0
        return cross_entropy

    def backward(self):
        return -self.y/(self.x+1e-15)

    def simplified_backward(self):
        return self.x - self.y
        #when combining softmax and cross-entropy this is what the gradient works out to