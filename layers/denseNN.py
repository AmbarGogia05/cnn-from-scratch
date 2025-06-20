import numpy as np

class DenseNNLayer:
    def __init__(self, input_features, output_features):
        self.in_features = input_features
        self.out_features = output_features

        # Initializing weights as zeros caused gradients to explode
        # self.weights = np.zeros((self.out_features, self.in_features+1))
        # self.weights = np.random.randn(self.out_features, self.in_features + 1) * 0.0001
        limit = 1 / np.sqrt(self.in_features)
        self.weights = np.random.uniform(-limit, limit, (self.out_features, self.in_features + 1))
    def forward(self, x):
        weights = self.weights
        x_augmented = np.concatenate(([1], x))
        self.x_augmented = x_augmented
        output = weights @ x_augmented
        return output
    
    # The gradient received from the subsequent layer is dout, a 1D array
    def backward(self, dout):
        self.grad_weights = dout[:, None] * self.x_augmented[None, :]
        #This is stored to apply gradient descent
        dx_aug = self.weights.T @ dout
        return dx_aug[1:]

    def apply_gradients(self, lr):
        np.clip(self.grad_weights, -1, 1, out=self.grad_weights)
        self.weights -= lr * self.grad_weights
