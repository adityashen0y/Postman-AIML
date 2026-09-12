import sys
import numpy as np
import torch
from nn_from_scratch import NeuralNetwork

def run_correctness_harness(tolerance: float = 1e-5) -> bool:
    """
    Compares manual NumPy gradients against PyTorch torch.autograd.
    Reports explicit PASS/FAIL.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size, input_dim, hidden_dim, output_dim = 32, 8, 12, 4
    
    # Generate synthetic input and targets
    X_np = np.random.randn(batch_size, input_dim)
    y_labels = np.random.randint(0, output_dim, size=(batch_size,))
    Y_onehot = np.eye(output_dim)[y_labels]

    # Initialize NumPy Model and compute gradients
    model = NeuralNetwork(input_dim, hidden_dim, output_dim)
    _ = model.forward(X_np)
    model.backward(Y_onehot)

    # Convert to PyTorch Tensors for Autograd Reference
    X_t = torch.tensor(X_np, dtype=torch.float32)
    Y_t = torch.tensor(y_labels, dtype=torch.long)

    W1_t = torch.tensor(model.W1, dtype=torch.float32, requires_grad=True)
    b1_t = torch.tensor(model.b1, dtype=torch.float32, requires_grad=True)
    W2_t = torch.tensor(model.W2, dtype=torch.float32, requires_grad=True)
    b2_t = torch.tensor(model.b2, dtype=torch.float32, requires_grad=True)

    # PyTorch Forward Pass
    Z1_t = torch.matmul(X_t, W1_t) + b1_t
    A1_t = torch.relu(Z1_t)
    Z2_t = torch.matmul(A1_t, W2_t) + b2_t
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_t = loss_fn(Z2_t, Y_t)
    
    # PyTorch Backward Pass
    loss_t.backward()

    # Calculate absolute differences
    diffs = {
        "W1": np.max(np.abs(model.dW1 - W1_t.grad.numpy())),
        "b1": np.max(np.abs(model.db1 - b1_t.grad.numpy())),
        "W2": np.max(np.abs(model.dW2 - W2_t.grad.numpy())),
        "b2": np.max(np.abs(model.db2 - b2_t.grad.numpy())),
    }

    print("=" * 50)
    print("       CORRECTNESS HARNESS GRADIENT CHECK        ")
    print("=" * 50)
    all_passed = True
    for param_name, max_diff in diffs.items():
        passed = max_diff < tolerance
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"Parameter [{param_name:>2}] | Max Absolute Diff: {max_diff:.8e} | Status: [{status}]")

    print("-" * 50)
    if all_passed:
        print("OVERALL RESULT: [PASS] All manual gradients match PyTorch autograd!")
        print("=" * 50)
        return True
    else:
        print("OVERALL RESULT: [FAIL] Gradient mismatch exceeds tolerance threshold!")
        print("=" * 50)
        return False

if __name__ == "__main__":
    success = run_correctness_harness()
    sys.exit(0 if success else 1)