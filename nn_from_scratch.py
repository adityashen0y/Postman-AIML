import numpy as np

class NeuralNetwork:
    """
    2-Layer Feedforward Neural Network implemented with NumPy.
    Architecture: Input -> Linear -> ReLU -> Linear -> Softmax
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        # He initialization for ReLU hidden layer
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        
        # Xavier/Glorot initialization for Softmax layer
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b2 = np.zeros((1, output_dim))

    def relu(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)

    def relu_derivative(self, Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(float)

    def softmax(self, Z: np.ndarray) -> np.ndarray:
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, Y_onehot: np.ndarray):
        """
        Computes gradients via manual backpropagation.
        """
        m = self.X.shape[0]

        # Output layer gradient (Categorical Cross-Entropy + Softmax derivative)
        self.dZ2 = self.A2 - Y_onehot
        self.dW2 = np.dot(self.A1.T, self.dZ2) / m
        self.db2 = np.sum(self.dZ2, axis=0, keepdims=True) / m

        # Hidden layer backpropagation
        self.dA1 = np.dot(self.dZ2, self.W2.T)
        self.dZ1 = self.dA1 * self.relu_derivative(self.Z1)
        self.dW1 = np.dot(self.X.T, self.dZ1) / m
        self.db1 = np.sum(self.dZ1, axis=0, keepdims=True) / m

    def step(self, lr: float):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2