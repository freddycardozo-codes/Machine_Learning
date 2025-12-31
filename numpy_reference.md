# NumPy Statements Reference Guide

This document provides a comprehensive reference of all NumPy operations used in the `delivery_time_nn.py` neural network implementation.

---

## Table of Contents
1. [Array Creation](#array-creation)
2. [Random Number Generation](#random-number-generation)
3. [Mathematical Operations](#mathematical-operations)
4. [Array Manipulation](#array-manipulation)
5. [Aggregation Functions](#aggregation-functions)
6. [Linear Algebra](#linear-algebra)
7. [Comparison Operations](#comparison-operations)

---

## Array Creation

| Statement | Description | Output |
|-----------|-------------|--------|
| `np.zeros((1, hidden_size))` | Creates an array filled with zeros of specified shape. Used to initialize bias vectors. | `array([[0., 0., 0., 0., 0., 0., 0., 0.]])` for shape (1, 8) |
| `np.array([[dist_norm]])` | Creates a NumPy array from a Python nested list. Used to convert scalar input to 2D array for network input. | `array([[0.5]])` if dist_norm=0.5 |

---

## Random Number Generation

| Statement | Description | Output |
|-----------|-------------|--------|
| `np.random.seed(42)` | Sets the random seed for reproducibility. All subsequent random operations will produce the same sequence. | `None` (modifies global state) |
| `np.random.randn(input_size, hidden_size)` | Generates random samples from standard normal distribution (mean=0, std=1). Used for weight initialization. | `array([[-0.41, 0.59, ...]])` shape (input_size, hidden_size) |
| `np.random.uniform(1, 50, n_samples)` | Generates random samples from uniform distribution between low (1) and high (50). Used to create random distances. | `array([38.34, 12.57, 4.83, ...])` with n_samples values |
| `np.random.normal(0, noise * distances, shape)` | Generates random samples from normal (Gaussian) distribution with specified mean and std. Used to add realistic noise. | `array([[0.12], [-0.45], ...])` noise values |

---

## Mathematical Operations

| Statement | Description | Output |
|-----------|-------------|--------|
| `np.sqrt(2.0 / input_size)` | Computes the square root. Used in He initialization to scale weights based on layer size. | `1.414` for input_size=1 |
| `np.maximum(0, x)` | Element-wise maximum comparison. Returns x where x > 0, else 0. Implements ReLU activation. | `array([[0, 2, 0, 5]])` for input `[[-1, 2, -3, 5]]` |
| `(y_pred - y_true) ** 2` | Element-wise subtraction followed by squaring. Computes squared errors for MSE loss. | `array([[0.04], [0.09]])` for errors [0.2, 0.3] |
| `data * (max_val - min_val) + min_val` | Element-wise multiplication and addition. Reverses min-max normalization (denormalize). | Original scale values |
| `(data - min_val) / (max_val - min_val)` | Element-wise subtraction and division. Applies min-max normalization to [0, 1] range. | Values in range [0, 1] |
| `da1 * self.relu_derivative(self.z1)` | Element-wise multiplication of two arrays. Used in backpropagation to apply chain rule. | Gradient array of same shape |
| `self.W2 -= self.learning_rate * dW2` | In-place subtraction (compound assignment). Updates weights using gradient descent. | Modified weight matrix |
| `noise * distances` | Element-wise multiplication (broadcasting). Scales noise proportionally to distance. | `array([[1.5], [3.0], ...])` for noise=0.15 |

---

## Array Manipulation

| Statement | Description | Output |
|-----------|-------------|--------|
| `.reshape(-1, 1)` | Reshapes array to specified dimensions. -1 means "infer this dimension". Converts 1D to 2D column vector. | `array([[1], [2], [3]])` from `[1, 2, 3]` |
| `.flatten()` | Collapses array to 1D. Used before argsort for indexing operations. | `array([1, 2, 3])` from `[[1], [2], [3]]` |
| `self.a1.T` | Transposes the array (swaps rows and columns). Used in matrix multiplication during backprop. | Shape (8, n_samples) from (n_samples, 8) |
| `self.W2.T` | Transposes weight matrix. Used to propagate gradients backward through layers. | Shape (1, 8) from (8, 1) |
| `X_norm[:split_idx]` | Array slicing. Selects elements from start to split_idx (exclusive). Creates training set. | First 80% of samples |
| `X_norm[split_idx:]` | Array slicing. Selects elements from split_idx to end. Creates test set. | Last 20% of samples |
| `distances[sorted_idx]` | Fancy indexing. Reorders array elements based on index array. Used for plotting sorted data. | Sorted array |
| `pred_norm[0, 0]` | Multi-dimensional indexing. Extracts scalar value from 2D array at row 0, column 0. | Single float value |

---

## Aggregation Functions

| Statement | Description | Output |
|-----------|-------------|--------|
| `np.mean((y_pred - y_true) ** 2)` | Computes arithmetic mean of all elements. Calculates MSE loss across all samples. | `0.0013` (single scalar) |
| `np.sum(dz2, axis=0, keepdims=True)` | Sums array elements along axis 0 (rows). keepdims=True preserves 2D shape for broadcasting. | `array([[sum_value]])` shape (1, 1) |
| `np.sum(dz1, axis=0, keepdims=True)` | Sums gradients across all samples. Used to compute bias gradients in hidden layer. | `array([[s1, s2, ..., s8]])` shape (1, 8) |
| `data.min()` | Returns minimum value in the array. Used in normalization to find range. | Single minimum value |
| `data.max()` | Returns maximum value in the array. Used in normalization to find range. | Single maximum value |
| `np.mean(np.abs(y_pred_denorm - delivery_times))` | Computes Mean Absolute Error. Average of absolute differences. | `2.72` (minutes) |

---

## Linear Algebra

| Statement | Description | Output |
|-----------|-------------|--------|
| `np.dot(X, self.W1)` | Matrix multiplication. Computes weighted sum for hidden layer: (n_samples, 1) @ (1, 8) = (n_samples, 8). | Weighted inputs for hidden layer |
| `np.dot(self.a1, self.W2)` | Matrix multiplication. Computes weighted sum for output layer: (n_samples, 8) @ (8, 1) = (n_samples, 1). | Network output predictions |
| `np.dot(self.a1.T, dz2)` | Matrix multiplication for gradient computation. Computes dW2: (8, n_samples) @ (n_samples, 1) = (8, 1). | Gradient for W2 weights |
| `np.dot(dz2, self.W2.T)` | Matrix multiplication. Propagates error back to hidden layer: (n_samples, 1) @ (1, 8) = (n_samples, 8). | Hidden layer error signal |
| `np.dot(X.T, dz1)` | Matrix multiplication for gradient computation. Computes dW1: (1, n_samples) @ (n_samples, 8) = (1, 8). | Gradient for W1 weights |

---

## Comparison Operations

| Statement | Description | Output |
|-----------|-------------|--------|
| `(x > 0)` | Element-wise comparison. Creates boolean array where True indicates positive values. | `array([[False, True, False, True]])` |
| `(x > 0).astype(float)` | Type conversion. Converts boolean array to float (True→1.0, False→0.0). Implements ReLU derivative. | `array([[0., 1., 0., 1.]])` |

---

## Sorting and Indexing

| Statement | Description | Output |
|-----------|-------------|--------|
| `np.argsort(distances.flatten())` | Returns indices that would sort the array. Used to order data points by distance for plotting. | `array([2, 0, 3, 1, ...])` index array |

---

## Summary Table: All NumPy Operations by Category

| Category | Functions Used |
|----------|---------------|
| **Array Creation** | `np.zeros()`, `np.array()` |
| **Random** | `np.random.seed()`, `np.random.randn()`, `np.random.uniform()`, `np.random.normal()` |
| **Math** | `np.sqrt()`, `np.maximum()`, `np.abs()`, `**` (power), `*`, `-`, `/`, `+` |
| **Manipulation** | `.reshape()`, `.flatten()`, `.T` (transpose), slicing `[:]`, indexing `[i,j]` |
| **Aggregation** | `np.mean()`, `np.sum()`, `.min()`, `.max()` |
| **Linear Algebra** | `np.dot()` |
| **Comparison** | `>`, `.astype()` |
| **Sorting** | `np.argsort()` |

---

## Usage Examples in Neural Network Context

### Weight Initialization (He Initialization)
```python
# Creates random weights scaled for ReLU activation
W1 = np.random.randn(1, 8) * np.sqrt(2.0 / 1)
# Output: array([[-0.58, 1.23, -0.89, 0.45, 1.67, -0.34, 0.92, -1.12]])
```

### Forward Propagation
```python
# Hidden layer: z = X @ W + b, then apply ReLU
z1 = np.dot(X, W1) + b1      # Linear transformation
a1 = np.maximum(0, z1)        # ReLU activation
# Output shapes: z1=(160, 8), a1=(160, 8)
```

### Loss Computation (MSE)
```python
# Mean Squared Error
loss = np.mean((y_pred - y_true) ** 2)
# Output: 0.0013 (scalar)
```

### Gradient Computation
```python
# Compute gradients for backpropagation
dW2 = (1/m) * np.dot(a1.T, dz2)
db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
# Output shapes: dW2=(8, 1), db2=(1, 1)
```

### Data Normalization
```python
# Min-max normalization to [0, 1]
X_norm = (X - X.min()) / (X.max() - X.min())
# Output: values in range [0, 1]
```

---

## Key Concepts

### Broadcasting
NumPy automatically expands smaller arrays to match larger ones:
```python
z1 = np.dot(X, W1) + b1
# X @ W1 shape: (160, 8)
# b1 shape: (1, 8)
# Broadcasting adds b1 to each row of the result
```

### Vectorization
Operations apply to entire arrays without explicit loops:
```python
# Instead of: for i in range(n): result[i] = max(0, x[i])
result = np.maximum(0, x)  # Vectorized - much faster!
```

### Matrix Dimensions in Neural Networks
| Operation | Input Shape | Weight Shape | Output Shape |
|-----------|-------------|--------------|--------------|
| Hidden Layer | (n, 1) | (1, 8) | (n, 8) |
| Output Layer | (n, 8) | (8, 1) | (n, 1) |

---

*This reference is specific to the `delivery_time_nn.py` implementation.*

