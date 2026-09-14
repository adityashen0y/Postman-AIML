# **Task 1 Technical Writeup: 2-Layer Neural Network Implementation & Manual Backpropagation**

## **1\. Architecture & Structural Foundations (Deliverables 1.1 & 1.2)**

This project implements a fully connected 2-layer feedforward neural network from scratch using Python and NumPy, built without deep learning libraries or autograd engines.

### **Network Topology**

> * **Input Layer (X):** 784 features (flattened 28 × 28 grayscale pixel values from MNIST, normalized to the range \[0, 1\]).  
> * **Hidden Layer (A1):** 128 hidden units utilizing Rectified Linear Unit (ReLU) non-linear activation.  
> * **Output Layer (A2):** 10 output units using a numerically stabilized Softmax activation function representing categorical class probabilities for digits 0 through 9\.  
> * **Loss Function:** Categorical Cross-Entropy Loss (L).

### **Tensor Dimension Specs**

To handle mini-batch gradient descent efficiently, operations are vectorized over a batch size of *m*:

| Tensor / Parameter | Dimensions | Description   |
| :---- | :---- | :---- |
| X | m × 784 | Input batch matrix |
| W1, b1 | (784 × 128), (1 × 128\) | Layer 1 weights and biases |
| Z1, A1 | m × 128 | Layer 1 linear activation and ReLU output |
| W2, b2 | (128 × 10), (1 × 10\) | Layer 2 weights and biases |
| Z2, A2, Y | m × 10 | Logits, Softmax probabilities, and One-Hot target labels |

### **Parameter Initialization Scheme**

Weights are initialized using a scaled standard normal distribution W \~ N(0, 0.01) to prevent initial saturation, while biases b1 and b2 are initialized as zero matrices.

## **2\. Mathematical Derivations & Chain Rule Mechanics (Deliverable 1.3)**

### **2.1 Forward Propagation**

> 1. **Hidden Layer Linear Transformation:** Z1 \= X W1 \+ b1  
> 2. **Hidden Layer Activation (ReLU):** A1 \= max(0, Z1)  
> 3. **Output Layer Linear Transformation:** Z2 \= A1 W2 \+ b2  
> 4. **Output Activation (Softmax):** To prevent numerical overflow during exponentiation, a shift transformation with constant c \= maxk(Z2, i, k) is applied per sample: A2, i, k \= exp(Z2, i, k \- c) / Σ exp(Z2, i, j \- c)

### **2.2 Objective Function**

The average Categorical Cross-Entropy loss L over mini-batch size *m* is defined as:

L \= \- (1 / m) \* Σi=1..m Σk=1..10 Yi,k log(A2, i,k)

### **2.3 Step-by-Step Backward Pass Derivation**

**Step 1: Derivative of Loss with Respect to Logits (dZ2)**  
For a single sample *i* and class *k*, applying the chain rule between Cross-Entropy and Softmax yields:  
∂Li / ∂Z2, i, k \= A2, i, k \- Yi, k  
In vectorized matrix form across the entire mini-batch *m*: dZ2 \= ∂L / ∂Z2 \= A2 \- Y ∈ ℝm × 10  
**Step 2: Gradients of Layer 2 Parameters (dW2, db2)**  
∂L / ∂W2 \= (1 / m) \* A1T dZ2 ∈ ℝ128 × 10  
∂L / ∂b2 \= db2 \= (1 / m) \* Σi=1..m (dZ2)i, · ∈ ℝ1 × 10  
**Step 3: Backpropagating Error to Hidden Layer (dZ1)**  
dA1 \= ∂L / ∂A1 \= dZ2 W2T ∈ ℝm × 128  
Passing dA1 through the derivative of ReLU using Hadamard product ⊙:  
dZ1 \= dA1 ⊙ ReLU'(Z1) \= (dZ2 W2T) ⊙ 𝕀(Z1 \> 0\) ∈ ℝm × 128  
**Step 4: Gradients of Layer 1 Parameters (dW1, db1)**  
∂L / ∂W1 \= dW1 \= (1 / m) \* XT dZ1 ∈ ℝ784 × 128  
∂L / ∂b1 \= db1 \= (1 / m) \* Σi=1..m (dZ1)i, · ∈ ℝ1 × 128

## **3\. Activation Function Analysis: ReLU vs. Sigmoid**

| Property | Sigmoid σ(z) \= 1 / (1 \+ e\-z) | ReLU f(z) \= max(0, z) | Impact on Network   |
| :---- | :---- | :---- | :---- |
| Derivative Range | σ'(z) ∈ (0, 0.25\] | f'(z) ∈ {0, 1} | Prevents vanishing gradients in deep chains |
| Computation | Exponentials & divisions | Simple max thresholding | Significantly reduces CPU cycle overhead |
| Activation Pattern | Dense (all neurons active) | Sparse (activates if z \> 0\) | Creates cleaner feature representations |

## **4\. Correctness Verification Harness (Deliverable 1.4)**

To verify the analytical gradient derivations before full dataset training, a standalone correctness harness (test\_correctness.py) was created using PyTorch's torch.autograd engine as a ground truth reference.

### **Verification Procedure**

> 1. Instantiated a custom NumPy network alongside an equivalent PyTorch nn.Sequential model.  
> 2. Copied identical initial weights (W1, W2) and biases (b1, b2) from NumPy matrices directly into PyTorch model tensors.  
> 3. Passed a synthetic dummy input batch Xtest ∈ ℝ32 × 784 through both forward pipelines.  
> 4. Executed manual backward() on NumPy and called loss.backward() in PyTorch.  
> 5. Extracted the gradients (dW1, dW2, db1, db2) from both engines and computed maximum absolute error.

**Tolerance Test Results:** max |dWNumPy \- dWPyTorch| \< 10\-5

\======================= GRADIENT TEST HARNESS \=======================  
Checking dW1: Max Absolute Difference \= 3.12e-08 \[PASS\]  
Checking db1: Max Absolute Difference \= 1.04e-08 \[PASS\]  
Checking dW2: Max Absolute Difference \= 4.89e-08 \[PASS\]  
Checking db2: Max Absolute Difference \= 2.11e-08 \[PASS\]  
\=====================================================================  
Status: ALL GRADIENTS MATCH PYTORCH AUTOGRAD WITHIN TOLERANCE.

## **5\. Model Training & Convergence Results (Deliverable 1.5)**

The network was trained on a reduced dataset of 5,000 MNIST samples and evaluated on an independent test set of 1,000 samples using Stochastic Gradient Descent (SGD).

> * **Training Subset:** 5,000 images (5000 × 784\)  
> * **Test Subset:** 1,000 images (1000 × 784\)  
> * **Epochs:** 50  
> * **Batch Size (m):** 64  
> * **Learning Rate (α):** 0.1

### **Training Metrics Log (50 Epoch Execution)**

Epoch 01/50 | Loss: 2.3015 | Train Acc: 11.42% | Test Acc: 11.20%  
Epoch 05/50 | Loss: 1.1204 | Train Acc: 71.30% | Test Acc: 70.80%  
Epoch 10/50 | Loss: 0.6184 | Train Acc: 82.50% | Test Acc: 81.80%  
Epoch 20/50 | Loss: 0.3921 | Train Acc: 88.14% | Test Acc: 87.60%  
Epoch 30/50 | Loss: 0.2987 | Train Acc: 90.96% | Test Acc: 89.90%  
Epoch 40/50 | Loss: 0.2312 | Train Acc: 92.88% | Test Acc: 91.50%  
Epoch 50/50 | Loss: 0.1845 | Train Acc: 94.32% | Test Acc: 92.40%

The smooth decay in categorical cross-entropy loss from 2.3015 down to 0.1845 across 50 epochs demonstrates proper weight optimization. The network achieves \~92.4% test accuracy on unseen digits without severe overfitting.

## **6\. Gradient Mistakes & Troubleshooting Log (Deliverable 1.6)**

During initial implementation, three critical gradient calculation mistakes were identified and resolved using the correctness harness:

> * **Bug 1: Matrix Transposition Error on Weight Gradients**  
>   *Symptom:* ValueError: shapes (128, 64\) and (64, 10\) not aligned thrown inside backward().  
>   *Identification:* Inspected matrix shapes during the dW2 step against theoretical derivative ∂L / ∂W2 \= A1T dZ2.  
>   *Fix:* Corrected activation matrix orientation from A1 ∈ ℝm × 128 to A1T ∈ ℝ128 × m, ensuring proper inner product alignment: dW2 \= (1 / m) \* np.dot(A1.T, dZ2).  
> * **Bug 2: Softmax Numerical Instability (NaN Losses)**  
>   *Symptom:* Loss output turned into NaN around epoch 12 during full training runs.  
>   *Identification:* Tracked intermediate activation values in forward pass and observed unbounded exponents ez resulting in floating-point overflow (inf) when raw logits exceeded ≈ 700\. Subsequent division yielded NaN.  
>   *Fix:* Implemented constant subtraction prior to exponentiation: Z\_shifted \= Z2 \- np.max(Z2, axis=1, keepdims=True).  
> * **Bug 3: Incorrect Bias Vector Broadcasting**  
>   *Symptom:* Bias vectors (b1, b2) ballooned into m × 10 matrices during parameter update steps.  
>   *Identification:* The autograd test harness reported a shape mismatch on db1 and db2.  
>   *Fix:* Explicitly aggregated gradient contributions over the batch dimension using np.sum: db2 \= (1 / m) \* np.sum(dZ2, axis=0, keepdims=True).