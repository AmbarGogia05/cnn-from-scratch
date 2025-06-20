

import numpy as np
from layers.conv2d import Conv
from layers.denseNN import DenseNNLayer
from layers.flatten import Flatten
from layers.relu import ReLU
from layers.cross_entropy import CrossEntropy

def test_minimal_model_backward_pass():
    # Create dummy input and label
    x = np.random.rand(1, 32, 32).astype(np.float32)
    y_true = np.zeros(10, dtype=np.float32)
    y_true[3] = 1  # dummy class

    # Build a minimal model
    model = [
        Conv(in_channels=1, out_channels=2, kernel_size=5, stride=1, padding=0),
        ReLU(),
        Flatten(),
        DenseNNLayer(input_features=2 * 28 * 28, output_features=10),
    ]

    loss_fn = CrossEntropy()

    # Forward pass
    out = x
    for layer in model:
        out = layer.forward(out)
        assert np.all(np.isfinite(out)), f"NaN or Inf detected after {layer.__class__.__name__} forward pass"

    loss = loss_fn.forward(out, y_true)

    # Backward pass
    dout = loss_fn.simplified_backward()
    for layer in reversed(model):
        dout = layer.backward(dout)

    # Apply gradients (simulate training step)
    for layer in model:
        if hasattr(layer, "apply_gradients"):
            layer.apply_gradients(0.01)

    # Check for NaNs/Infs in final output and gradients
    assert np.all(np.isfinite(out)), "NaN or Inf in final output"
    for layer in model:
        if hasattr(layer, "weights"):
            assert np.all(np.isfinite(layer.weights)), "NaN or Inf in weights"
        if hasattr(layer, "grad_weights"):
            assert np.all(np.isfinite(layer.grad_weights)), "NaN or Inf in grad_weights"

    print("✅ Backward pass test passed without NaNs or Infs.")

if __name__ == "__main__":
    test_minimal_model_backward_pass()