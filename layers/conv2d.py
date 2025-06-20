import numpy as np

class Conv:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        # In this implementation we have assumed a square filter with both dimensions equalling kernel_size
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)*0.001
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        # First, we pad the input to maintain consistency in the input and output dimensions after applying the convolutional filter
        pad = self.padding
        stride = self.stride
        k = self.kernel_size
        weights = self.weights
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
        self.x = x
        dimx = x.shape[1]
        dimy = x.shape[2]
        # We initialize the output matrix where we store the product values
        out_channels = self.out_channels
        out_height = (dimx - k) // stride + 1
        out_width = (dimy - k) // stride + 1
        output = np.zeros((out_channels, out_height, out_width))
        
        # We make slices of x, for which we use step size = stride and slice width as kernel size
        # Think about the range for i and j while slicing, it is crucial to this step
        for i in range (0, dimx-k+1, stride):
            for j in range(0, dimy-k+1, stride):
                patch = x[:, i:i+k, j:j+k]
                for channel in range(out_channels):
                    # To apply a convolution, we multiply element-wise each value of the filter matrix with the slice, and also add the bias afterwards
                    prod = weights[channel, :, :, :] * patch
                    val = np.sum(prod) + self.bias[channel]
                    output[channel, i//stride, j//stride] = val
        return output
    
    def backward(self, dout):
        self.grad_bias = np.sum(dout, axis=(1, 2))

        self.grad_weights = np.zeros_like(self.weights)

        # Unpack
        k = self.kernel_size
        stride = self.stride
        x = self.x  # padded input
        out_channels, out_height, out_width = dout.shape

        # Loop over spatial positions
        for i in range(0, out_height):
            for j in range(0, out_width):
                # Get the corresponding input patch
                i_start = i*stride
                j_start = j*stride
                patch = x[:, i_start:i_start + k, j_start:j_start + k]  # shape: (C_in, K, K)

                # For each output channel
                for c in range(out_channels):
                    # Multiply patch with dout[c, i, j] and accumulate
                    self.grad_weights[c] += patch * dout[c, i, j]

                # Initialize dx
        dx = np.zeros_like(self.x)

        # Loop over spatial positions in dout
        for i in range(out_height):
            for j in range(out_width):
                i_start = i*stride
                j_start = j*stride

                for c in range(out_channels):
                    # The filter for output channel c
                    filt = self.weights[c]  # shape: (C_in, K, K)
                    # Distribute the dout value backward over the input slice
                    dx[:, i_start:i_start+k, j_start:j_start+k] += dout[c, i, j] * np.flip(filt, axis=(1,2))

        if self.padding > 0:
            return dx[:, self.padding:-self.padding, self.padding:-self.padding]
        else:
            return dx
    def apply_gradients(self, lr):
        np.clip(self.grad_weights, -1, 1, out=self.grad_weights)
        np.clip(self.grad_bias, -1, 1, out=self.grad_bias)
        self.weights -= lr * self.grad_weights
        self.bias -= lr * self.grad_bias