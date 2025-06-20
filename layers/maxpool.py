import numpy as np
# Here, we maxpool assuming a 3D array x
class MaxPool:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        kernel_size = self.kernel_size
        stride = self.stride
        width = x.shape[-1]
        height = x.shape[-2]
        output_height = (height - kernel_size) // stride + 1
        output_width = (width - kernel_size) // stride + 1
        output = np.zeros((x.shape[0], output_height, output_width))
        self.max_mask = np.zeros_like(x)
        self.max_indices = np.zeros((x.shape[0], output_height, output_width, 2), dtype=int)
        for i in range(0, height - kernel_size + 1, stride):
            for j in range(0, width - kernel_size + 1, stride):
                for channel in range(x.shape[0]):
                    chunk = x[channel, i:i + kernel_size, j:j + kernel_size]
                    output[channel, i // stride, j // stride] = np.max(chunk)
                    flat_index = np.argmax(chunk)
                    i_max, j_max = np.unravel_index(flat_index, chunk.shape)
                    self.max_mask[channel, i + i_max, j + j_max] = 1
                    self.max_indices[channel, i // stride, j // stride] = (i + i_max, j + j_max)
        return output

    def backward(self, dout):
        dx = np.zeros_like(self.max_mask)
        C, out_H, out_W = dout.shape
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    i_max, j_max = self.max_indices[c, i, j]
                    dx[c, i_max, j_max] = dout[c, i, j]
        return dx