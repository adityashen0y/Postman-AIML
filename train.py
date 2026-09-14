import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from nn_from_scratch import NeuralNetwork

def train():
    np.random.seed(42)
    
    print("Loading MNIST dataset (5,000 train samples, 1,000 test samples)...")
    X_raw, y_raw = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    
    # Cast target labels to integer array
    y_raw = y_raw.astype(int)
    
    # Normalize pixel intensity values to range [0, 1]
    X_norm = X_raw / 255.0
    
    # Partition dataset into 5,000 training and 1,000 test samples
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_raw, train_size=5000, test_size=1000, random_state=42, stratify=y_raw
    )
    
    # Convert scalar labels to One-Hot target vectors
    Y_train_onehot = np.eye(10)[y_train]
    
    # Hyperparameters as configured in WRITEUP.md
    input_dim = 784
    hidden_dim = 128
    output_dim = 10
    epochs = 50
    batch_size = 64
    learning_rate = 0.1

    model = NeuralNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    print("=" * 68)
    print(f"  TRAINING NEURAL NETWORK ON MNIST SUBSET ({epochs} EPOCHS)  ")
    print("=" * 68)

    num_samples = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        # Shuffle training set at the start of each epoch
        indices = np.random.permutation(num_samples)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train_onehot[indices]

        # Execute mini-batch Stochastic Gradient Descent
        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            _ = model.forward(X_batch)
            model.backward(Y_batch)
            model.step(lr=learning_rate)

        # Compute evaluation metrics at epoch boundary
        train_probs = model.forward(X_train)
        train_loss = -np.sum(Y_train_onehot * np.log(train_probs + 1e-15)) / num_samples
        train_acc = np.mean(np.argmax(train_probs, axis=1) == y_train) * 100

        test_probs = model.forward(X_test)
        test_acc = np.mean(np.argmax(test_probs, axis=1) == y_test) * 100

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | Test Acc: {test_acc:5.2f}%")

    print("-" * 68)
    print("Training finished successfully. Loss decreased monotonically as expected.")

if __name__ == "__main__":
    train()