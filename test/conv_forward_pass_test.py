import numpy as np
from layers.conv2d import Conv  # Adjust import path based on your actual file structure

def test_conv_forward_pass():
    in_channels = 1
    out_channels = 1
    kernel_size = 3
    stride = 1
    padding = 0

    conv = Conv(in_channels, out_channels, kernel_size, stride, padding)

    # Use a fixed weight and bias to make the test deterministic
    conv.weights = np.ones((out_channels, in_channels, kernel_size, kernel_size))
    conv.bias = np.zeros(out_channels)

    # Input: 1 channel, 5x5 matrix
    x = np.array([
        [[1, 2, 3, 4, 5],
         [5, 6, 7, 8, 9],
         [9, 8, 7, 6, 5],
         [5, 4, 3, 2, 1],
         [1, 2, 3, 4, 5]]
    ])

    output = conv.forward(x)

    # Expected output size: (1, 3, 3)
    expected_output_shape = (out_channels, (5 - kernel_size)//stride + 1, (5 - kernel_size)//stride + 1)

    assert output.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, got {output.shape}"
    print("Output:\n", output)

if __name__ == "__main__":
    test_conv_forward_pass()
