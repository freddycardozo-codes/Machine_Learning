# Neural Network from Scratch - Delivery Time Prediction

A comprehensive educational project demonstrating how to build a neural network from scratch using only NumPy. This project predicts delivery times based on distance traveled, showcasing fundamental deep learning concepts.

## Overview

This project implements a feedforward neural network without using deep learning frameworks like PyTorch or TensorFlow. It's designed for learning purposes to understand the core mechanics of neural networks:

- **Forward Propagation**: How inputs flow through the network to produce predictions
- **Backpropagation**: How gradients are computed using the chain rule
- **Gradient Descent**: How weights are updated to minimize loss
- **Activation Functions**: ReLU implementation and its derivative
- **Data Normalization**: Min-max scaling for better training

## Architecture

```
Input Layer (1 neuron) → Hidden Layer (8 neurons, ReLU) → Output Layer (1 neuron)
```

- **Input**: Distance in kilometers
- **Output**: Predicted delivery time in minutes
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Batch Gradient Descent

## Project Structure

```
pytorch/
├── delivery_time_nn.py      # Main neural network implementation
├── numpy_random_examples.py # NumPy random functions tutorial
├── numpy_reference.md       # NumPy operations reference guide
├── requirements.txt         # Project dependencies
├── README.md               # This file
└── .gitignore              # Git ignore rules
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/neural-network-from-scratch.git
cd neural-network-from-scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Neural Network Training

```bash
python delivery_time_nn.py
```

This will:
1. Generate synthetic delivery data (200 samples)
2. Train the neural network for 1000 epochs
3. Evaluate on test set
4. Display sample predictions
5. Generate visualization (`delivery_nn_results.png`)

### Expected Output

```
============================================================
   Neural Network from Scratch - Delivery Time Prediction
============================================================

Generating delivery data...
   Training samples: 160
   Test samples: 40

Training the network...
Epoch 100/1000, Loss: 0.0022
Epoch 200/1000, Loss: 0.0014
...
Epoch 1000/1000, Loss: 0.0013

Test Loss (MSE): 0.000852
Mean Absolute Error: 2.72 minutes

Predicted delivery times:
Distance (km)   Predicted Time (min)
5               20.0
10              30.1
20              50.2
30              70.2
45              100.4
```

### Run NumPy Examples

```bash
python numpy_random_examples.py
```

Learn about `np.random.uniform` and `np.random.normal` with practical examples.

## Key Concepts Explained

### Forward Propagation
```python
# Hidden layer: linear transformation + ReLU activation
z1 = np.dot(X, W1) + b1
a1 = np.maximum(0, z1)  # ReLU

# Output layer: linear transformation (regression)
z2 = np.dot(a1, W2) + b2
output = z2
```

### Backpropagation
```python
# Compute gradients using chain rule
dz2 = output - y_true
dW2 = (1/m) * np.dot(a1.T, dz2)
da1 = np.dot(dz2, W2.T)
dz1 = da1 * (z1 > 0)  # ReLU derivative
dW1 = (1/m) * np.dot(X.T, dz1)
```

### Weight Update (Gradient Descent)
```python
W2 -= learning_rate * dW2
W1 -= learning_rate * dW1
```

## Results

The trained model achieves:
- **Test MSE**: ~0.0009
- **Mean Absolute Error**: ~2.7 minutes

The model learns the underlying relationship:
```
delivery_time ≈ 10 + 2 × distance (minutes)
```

## Visualization

![Training Results](delivery_nn_results.png)

- **Left Plot**: Actual data points vs. neural network predictions
- **Right Plot**: Training loss curve (MSE over epochs)

## Dependencies

- Python 3.8+
- NumPy >= 1.20.0
- Matplotlib >= 3.5.0 (optional, for visualization)

## Learning Resources

- `delivery_time_nn.py`: Fully commented implementation explaining each step
- `numpy_reference.md`: Comprehensive NumPy operations reference table
- `numpy_random_examples.py`: Interactive examples for random distributions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

Neural Network Tutorial - Educational Project

---

*This project is part of a deep learning tutorial series demonstrating neural network fundamentals.*

