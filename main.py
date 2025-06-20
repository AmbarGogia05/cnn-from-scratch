# Here, we will try to implement the original LeNet architecture as was proposed in the paper
from tensorflow.keras.datasets import mnist # type: ignore
import numpy as np
from layers.conv2d import Conv
from layers.maxpool import MaxPool
from layers.flatten import Flatten
from layers.denseNN import DenseNNLayer
from layers.relu import ReLU
from layers.softmax import SoftMax
from layers.cross_entropy import CrossEntropy

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X.astype(np.float32) / 255.0
test_X = test_X.astype(np.float32) / 255.0

train_X = train_X[:, None, :, :]  # adds channel dim
test_X = test_X[:, None, :, :]

# Pad to (1, 32, 32) to match LeNet
train_X = np.pad(train_X, ((0, 0), (0, 0), (2, 2), (2, 2)))
test_X = np.pad(test_X, ((0, 0), (0, 0), (2, 2), (2, 2)))

train_y_oh = np.eye(10)[train_y]
test_y_oh = np.eye(10)[test_y]


model = [
    Conv(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
    MaxPool(kernel_size=2, stride=2),
    Conv(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
    MaxPool(kernel_size=2, stride=2),
    Flatten(),
    DenseNNLayer(input_features=400, output_features=120),
    ReLU(),
    DenseNNLayer(input_features=120, output_features=84),
    ReLU(),
    DenseNNLayer(input_features=84, output_features=10),
    SoftMax()
]

loss_fn = CrossEntropy()
learning_rate = 0.01
num_epochs = 5
num_samples = train_X.shape[0]

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0

    for i in range(num_samples):
        x = train_X[i]
        y_true = train_y_oh[i]

        # Forward pass through the model
        out = x
        for layer in model:
            out = layer.forward(out)

        # Compute loss
        loss = loss_fn.forward(out, y_true)
        total_loss += loss

        # Accuracy check
        if np.argmax(out) == np.argmax(y_true):
            correct += 1

        # Backward pass
        dout = loss_fn.simplified_backward()
        for layer in reversed(model):
            dout = layer.backward(dout)

        # Apply gradients
        for layer in model:
            if hasattr(layer, "apply_gradients"):
                layer.apply_gradients(learning_rate)

    avg_loss = total_loss / num_samples
    accuracy = correct / num_samples
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

correct = 0
num_test = test_X.shape[0]

for i in range(num_test):
    x = test_X[i]
    y_true = test_y_oh[i]

    out = x
    for layer in model:
        out = layer.forward(out)

    if np.argmax(out) == np.argmax(y_true):
        correct += 1

accuracy = correct / num_test
print(f"Test Accuracy: {accuracy:.4f}")