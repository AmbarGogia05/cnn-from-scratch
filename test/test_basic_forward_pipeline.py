import numpy as np
from layers.conv2d import Conv
from layers.denseNN import DenseNNLayer
from layers.relu import relu
from layers.maxpool import maxpool
from layers.flatten import flatten


def test_full_pipeline():
    # Step 1: Create input
    x = np.random.randn(1, 6, 6)  # 1 channel, 6x6 input

    # Step 2: Conv layer
    conv = Conv(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)
    conv.weights = np.ones_like(conv.weights)  # Make output predictable
    conv.bias = np.zeros(conv.out_channels)
    x = conv.forward(x)

    # Step 3: ReLU
    x = relu(x)

    # Step 4: MaxPool
    x = maxpool(x, kernel_size=2, stride=2)

    # Step 5: Flatten
    x = flatten(x)

    # Step 6: Dense layer
    dense = DenseNNLayer(input_features=x.shape[0], output_features=4)
    dense.weights = np.ones_like(dense.weights)
    output = dense.forward(x)

    print("Final output from full pipeline:", output)

if __name__ == "__main__":
    test_full_pipeline()
