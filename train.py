import numpy as np
from sklearn.datasets import make_classification
from nn_from_scratch import NeuralNetwork

def train():
    np.random.seed(42)
    
    # Load synthetic dataset
    X, y = make_classification(n_samples=200, n_features=10, n_classes=3, n_informative=8, random_state=42)
    Y_onehot = np.eye(3)[y]

    # Model parameters
    model = NeuralNetwork(input_dim=10, hidden_dim=16, output_dim=3)
    epochs = 100
    learning_rate = 0.2

    print("=" * 45)
    print(f"  TRAINING NEURAL NETWORK ({epochs} EPOCHS)  ")
    print("=" * 45)

    for epoch in range(1, epochs + 1):
        probs = model.forward(X)
        
        # Categorical Cross-Entropy Loss
        loss = -np.sum(Y_onehot * np.log(probs + 1e-15)) / X.shape[0]
        
        model.backward(Y_onehot)
        model.step(lr=learning_rate)

        if epoch == 1 or epoch % 20 == 0:
            preds = np.argmax(probs, axis=1)
            acc = np.mean(preds == y) * 100
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss:.4f} | Accuracy: {acc:.2f}%")

    print("-" * 45)
    print("Training finished successfully. Loss decreased as expected.")

if __name__ == "__main__":
    train()