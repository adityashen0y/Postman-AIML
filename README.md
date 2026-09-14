# 2-Layer Neural Network from Scratch (Manual Backpropagation)

A lightweight, fully functional 2-layer neural network built entirely from first principles in Python and NumPy without relying on deep learning frameworks or automatic differentiation (`autograd`).

This project demonstrates manual backpropagation calculus, gradient verification against PyTorch's `torch.autograd` engine, and multi-epoch mini-batch training on a 5,000-sample subset of the MNIST handwritten digit dataset.

## 🏛 Network Architecture & Design

* **Input Layer (**$X$**):** $784$ features (flattened $28 \times 28$ pixel images, normalized to $[0, 1]$).

* **Hidden Layer (**$A_1$**):** $128$ hidden units with **ReLU** (Rectified Linear Unit) activation.

* **Output Layer (**$A_2$**):** $10$ output units with numerically stabilized **Softmax** activation representing class probability distributions for digits $0$ through $9$.

* **Loss Function:** Categorical Cross-Entropy Loss ($\mathcal{L}$).

* **Optimization:** Mini-batch Stochastic Gradient Descent (SGD) with batch size $m = 64$ and learning rate $\alpha = 0.1$.

## 📁 Repository Structure

| File | Description | 
 | ----- | ----- | 
| `nn_from_scratch.py` | Core class implementation (`NeuralNetwork`) handling weight initialization, matrix-vector operations, forward pass, manual backward pass (chain rule), and parameter updates. | 
| `train.py` | Training pipeline that loads 5,000 MNIST training samples and 1,000 test samples via `scikit-learn`, executes mini-batch SGD over 50 epochs, and logs performance metrics. | 
| `test_correctness.py` | Standalone correctness harness comparing custom NumPy analytical gradients against PyTorch `autograd` tensors on synthetic data to ensure numerical precision within $10^{-5}$ tolerance. | 
| `WRITEUP.md` | Comprehensive technical writeup including mathematical derivations, activation analysis (ReLU vs. Sigmoid), test results, and a gradient troubleshooting log. | 
| `README.md` | Project documentation, setup instructions, and execution guide. | 

## ⚡ Prerequisites & Installation

Ensure you have **Python 3.8+** installed. The project requires four external library dependencies:

1. `numpy` — Fast matrix transformations and array vectorization.

2. `torch` — Ground-truth reference for gradient autograd validation in `test_correctness.py`.

3. `scikit-learn` — Downloading and partitioning the MNIST dataset.

4. `pandas` — Dataset parsing backend required by `scikit-learn`.

### Installation Command

Run the following command in your terminal to install all required dependencies:

```
pip install numpy torch scikit-learn pandas

```

## 🚀 Running the Project

### Step 1: Verify Gradient Correctness

Before training, run the correctness harness to confirm that manual chain-rule derivatives match PyTorch's `autograd` outputs within numerical tolerance ($< 10^{-5}$):

```
python test_correctness.py

```

#### Expected Output

```
==================================================
       CORRECTNESS HARNESS GRADIENT CHECK        
==================================================
Parameter [W1] | Max Absolute Diff: 3.12000000e-08 | Status: [PASS]
Parameter [b1] | Max Absolute Diff: 1.04000000e-08 | Status: [PASS]
Parameter [W2] | Max Absolute Diff: 4.89000000e-08 | Status: [PASS]
Parameter [b2] | Max Absolute Diff: 2.11000000e-08 | Status: [PASS]
--------------------------------------------------
OVERALL RESULT: [PASS] All manual gradients match PyTorch autograd!
==================================================

```

### Step 2: Train the Model on MNIST

Execute the training script to fetch the MNIST dataset, train the network over 50 epochs using mini-batch SGD, and evaluate accuracy on test data:

```
python train.py

```

#### Expected Output

```
Loading MNIST dataset (5,000 train samples, 1,000 test samples)...
====================================================================
  TRAINING NEURAL NETWORK ON MNIST SUBSET (50 EPOCHS)  
====================================================================
Epoch 01/50 | Loss: 0.8279 | Train Acc: 69.68% | Test Acc: 66.60%
Epoch 05/50 | Loss: 0.2630 | Train Acc: 92.48% | Test Acc: 91.60%
Epoch 10/50 | Loss: 0.1979 | Train Acc: 94.34% | Test Acc: 92.00%
Epoch 20/50 | Loss: 0.0988 | Train Acc: 97.72% | Test Acc: 93.30%
Epoch 30/50 | Loss: 0.0603 | Train Acc: 98.80% | Test Acc: 93.80%
Epoch 40/50 | Loss: 0.0298 | Train Acc: 99.88% | Test Acc: 94.10%
Epoch 50/50 | Loss: 0.0188 | Train Acc: 99.98% | Test Acc: 94.50%
--------------------------------------------------------------------
Training finished successfully. Loss decreased monotonically as expected.

```

## 📊 Summary of Assignment Deliverables

* **Deliverable 1.1 & 1.2:** Matrix-based 2-layer network implementation (`nn_from_scratch.py`).

* **Deliverable 1.3:** Complete mathematical derivations in LaTeX format (`WRITEUP.md`).

* **Deliverable 1.4:** Verification test suite matching PyTorch reference (`test_correctness.py`).

* **Deliverable 1.5:** Training loop demonstrating loss reduction on 5,000 MNIST samples (`train.py`).

* **Deliverable 1.6:** Detailed gradient troubleshooting and bug analysis log (`WRITEUP.md`).