"""
Neural Network with Sigmoid Activation - Delivery Time Prediction
==================================================================
A neural network implementation using Sigmoid activation function
to predict delivery time based on distance traveled.

This demonstrates the differences between Sigmoid and ReLU activations.

Key Characteristics of Sigmoid:
- Output range: (0, 1) - bounded output
- Smooth gradient everywhere
- Can suffer from vanishing gradient problem
- Historically popular, now often replaced by ReLU

Author: Neural Network Tutorial
"""

import numpy as np

# Try to import matplotlib for visualization (optional)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available. Visualization will be skipped.")


class NeuralNetworkSigmoid:
    """
    A feedforward neural network with Sigmoid activation.
    
    Architecture:
        Input Layer:  1 neuron (distance)
        Hidden Layer: 8 neurons (Sigmoid activation)
        Output Layer: 1 neuron (delivery time)
    
    Sigmoid Function: σ(x) = 1 / (1 + e^(-x))
    - Squashes input to range (0, 1)
    - Smooth, differentiable everywhere
    - Derivative: σ(x) * (1 - σ(x))
    """
    
    def __init__(self, input_size=1, hidden_size=8, output_size=1, learning_rate=0.5):
        """
        Initialize the neural network with random weights.
        
        Args:
            input_size: Number of input features (default 1 for distance)
            hidden_size: Number of neurons in hidden layer (default 8)
            output_size: Number of output neurons (default 1 for delivery time)
            learning_rate: Step size for gradient descent (default 0.5)
                          Note: Sigmoid often needs higher learning rate than ReLU
        """
        # Store the learning rate as an instance variable
        # Sigmoid networks often benefit from higher learning rates
        # because gradients are bounded (max gradient = 0.25)
        self.learning_rate = learning_rate
        
        # Initialize weights using Xavier/Glorot initialization
        # Xavier initialization is better suited for Sigmoid than He initialization
        # Formula: weights ~ N(0, sqrt(2 / (fan_in + fan_out)))
        # Simplified: weights ~ N(0, sqrt(1 / fan_in))
        # This helps maintain variance of activations through layers
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        
        # Initialize biases for layer 1 as zeros
        # Shape: (1, hidden_size) = (1, 8) for broadcasting with batch inputs
        self.b1 = np.zeros((1, hidden_size))
        
        # Initialize weights for layer 2 (hidden -> output) using Xavier initialization
        # Shape: (hidden_size, output_size) = (8, 1) for our case
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        
        # Initialize biases for layer 2 as zeros
        # Shape: (1, output_size) = (1, 1) for our case
        self.b2 = np.zeros((1, output_size))
        
        # Initialize placeholders for intermediate values needed in backpropagation
        # z1: Pre-activation values of hidden layer (before Sigmoid)
        self.z1 = None
        # a1: Activation values of hidden layer (after Sigmoid)
        self.a1 = None
        # z2: Pre-activation values of output layer
        self.z2 = None
        # a2: Activation values of output layer (final prediction)
        self.a2 = None
    
    def sigmoid(self, x):
        """
        Sigmoid activation function.
        
        Formula: σ(x) = 1 / (1 + e^(-x))
        
        Properties:
        - Output range: (0, 1) - always positive, bounded
        - σ(0) = 0.5 (centered)
        - Smooth S-shaped curve
        - Large positive x → σ(x) ≈ 1
        - Large negative x → σ(x) ≈ 0
        
        Args:
            x: Input array of any shape
            
        Returns:
            Array of same shape with values in (0, 1)
        """
        # Clip x to prevent overflow in exp for very large negative values
        # np.clip limits values to range [-500, 500] for numerical stability
        x_clipped = np.clip(x, -500, 500)
        
        # Apply sigmoid formula: 1 / (1 + e^(-x))
        return 1.0 / (1.0 + np.exp(-x_clipped))
    
    def sigmoid_derivative(self, sigmoid_output):
        """
        Derivative of Sigmoid activation function.
        
        Formula: σ'(x) = σ(x) * (1 - σ(x))
        
        Note: This takes the OUTPUT of sigmoid (not the input x)
        because we already computed sigmoid(x) in forward pass.
        
        Properties:
        - Maximum value: 0.25 (when σ(x) = 0.5)
        - Always positive
        - Approaches 0 for very large or very small inputs
          (This is the "vanishing gradient" problem)
        
        Args:
            sigmoid_output: Output from sigmoid function (a1 in our case)
            
        Returns:
            Array of same shape with derivative values
        """
        # Derivative formula using the sigmoid output directly
        # If s = sigmoid(x), then sigmoid'(x) = s * (1 - s)
        return sigmoid_output * (1.0 - sigmoid_output)
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        This computes the network's prediction by passing input through each layer:
        Input -> Hidden Layer (with Sigmoid) -> Output Layer (linear)
        
        Args:
            X: Input features of shape (n_samples, input_size)
               In our case: (n_samples, 1) where each row is a distance value
            
        Returns:
            Predicted delivery time of shape (n_samples, output_size)
        """
        # HIDDEN LAYER COMPUTATION
        # Compute the weighted sum (linear transformation) for hidden layer
        # z1 = X * W1 + b1
        # np.dot performs matrix multiplication: (n_samples, 1) @ (1, 8) = (n_samples, 8)
        # b1 is broadcast across all samples (added to each row)
        self.z1 = np.dot(X, self.W1) + self.b1
        
        # Apply Sigmoid activation function
        # This squashes all values to (0, 1) range
        # Unlike ReLU, sigmoid introduces non-linearity while keeping all neurons active
        # a1 has shape (n_samples, 8) - one activation per hidden neuron
        self.a1 = self.sigmoid(self.z1)
        
        # OUTPUT LAYER COMPUTATION
        # Compute the weighted sum for output layer
        # z2 = a1 * W2 + b2
        # Matrix multiplication: (n_samples, 8) @ (8, 1) = (n_samples, 1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        
        # Use linear activation (identity function) for output layer
        # For regression problems, we want unbounded continuous output
        # a2 = z2 (no transformation applied)
        self.a2 = self.z2  # Linear activation for regression
        
        # Return the final prediction (delivery time estimate)
        return self.a2
    
    def backward(self, X, y):
        """
        Backpropagation to compute gradients and update weights.
        
        This implements the chain rule to compute how much each weight
        contributed to the prediction error, then adjusts weights accordingly.
        
        Key difference from ReLU: Sigmoid derivative uses the activation output
        
        Args:
            X: Input features of shape (n_samples, input_size)
            y: True labels (delivery times) of shape (n_samples, output_size)
        """
        # Get the number of training samples for averaging gradients
        m = X.shape[0]
        
        # ==================== OUTPUT LAYER GRADIENTS ====================
        
        # Compute error at output layer (derivative of MSE loss)
        # dz2 = d(Loss)/d(z2) = (a2 - y)
        # Shape: (n_samples, 1)
        dz2 = self.a2 - y
        
        # Compute gradient of loss with respect to W2
        # dW2 = (1/m) * a1^T @ dz2
        # Shape: (8, n_samples) @ (n_samples, 1) = (8, 1)
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        
        # Compute gradient of loss with respect to b2
        # Sum across all samples and divide by m for average
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # ==================== HIDDEN LAYER GRADIENTS ====================
        
        # Propagate error back to hidden layer
        # da1 = dz2 @ W2^T
        # Shape: (n_samples, 1) @ (1, 8) = (n_samples, 8)
        da1 = np.dot(dz2, self.W2.T)
        
        # Apply chain rule through Sigmoid activation
        # dz1 = da1 * sigmoid_derivative(a1)
        # Key difference: Sigmoid derivative uses the activation output (a1)
        # not the pre-activation (z1) like ReLU
        # Shape: (n_samples, 8)
        dz1 = da1 * self.sigmoid_derivative(self.a1)
        
        # Compute gradient of loss with respect to W1
        # dW1 = (1/m) * X^T @ dz1
        # Shape: (1, n_samples) @ (n_samples, 8) = (1, 8)
        dW1 = (1/m) * np.dot(X.T, dz1)
        
        # Compute gradient of loss with respect to b1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # ==================== WEIGHT UPDATES (GRADIENT DESCENT) ====================
        
        # Update weights and biases using gradient descent rule
        # Note: Sigmoid networks often need higher learning rates
        
        # Update output layer weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        
        # Update hidden layer weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def compute_loss(self, y_pred, y_true):
        """
        Compute Mean Squared Error (MSE) loss.
        
        Args:
            y_pred: Predicted values from the network
            y_true: Actual target values
            
        Returns:
            Scalar MSE loss value
        """
        return np.mean((y_pred - y_true) ** 2)
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network using batch gradient descent.
        
        Args:
            X: Training features of shape (n_samples, input_size)
            y: Training labels of shape (n_samples, output_size)
            epochs: Number of training iterations (default 1000)
            verbose: If True, print progress every 100 epochs
            
        Returns:
            List of loss values at each epoch
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y)
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """Make predictions for new data."""
        return self.forward(X)


def generate_delivery_data(n_samples=100, noise=0.1):
    """
    Generate synthetic delivery data for training.
    
    Args:
        n_samples: Number of data points to generate
        noise: Amount of random variation
        
    Returns:
        Tuple of (distances, delivery_times)
    """
    np.random.seed(42)
    
    # Generate random distances (1 to 50 km)
    distances = np.random.uniform(1, 50, n_samples).reshape(-1, 1)
    
    # Delivery time formula:
    # Base preparation time: 10 minutes
    # Travel time: 2 minutes per km
    base_time = 10
    time_per_km = 2
    
    delivery_times = (
        base_time + 
        time_per_km * distances + 
        np.random.normal(0, noise * distances, (n_samples, 1))
    )
    
    return distances, delivery_times


def normalize(data, min_val=None, max_val=None):
    """Min-max normalization to [0, 1] range."""
    if min_val is None:
        min_val = data.min()
    if max_val is None:
        max_val = data.max()
    return (data - min_val) / (max_val - min_val), min_val, max_val


def denormalize(data, min_val, max_val):
    """Reverse min-max normalization."""
    return data * (max_val - min_val) + min_val


def plot_sigmoid_function():
    """Plot the sigmoid function and its derivative for visualization."""
    if not HAS_MATPLOTLIB:
        return
    
    x = np.linspace(-6, 6, 100)
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_deriv = sigmoid * (1 - sigmoid)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot sigmoid
    ax1 = axes[0]
    ax1.plot(x, sigmoid, 'b-', linewidth=2, label='Sigmoid: σ(x)')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('σ(x)', fontsize=12)
    ax1.set_title('Sigmoid Function', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # Plot derivative
    ax2 = axes[1]
    ax2.plot(x, sigmoid_deriv, 'r-', linewidth=2, label="Sigmoid': σ(x)(1-σ(x))")
    ax2.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Max = 0.25')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel("σ'(x)", fontsize=12)
    ax2.set_title('Sigmoid Derivative', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 0.3)
    
    plt.tight_layout()
    plt.savefig('sigmoid_function.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_results(distances, actual_times, predicted_times, losses):
    """Visualize the training results."""
    if not HAS_MATPLOTLIB:
        print("   (Skipping plot - matplotlib not available)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Predictions vs Actual
    ax1 = axes[0]
    ax1.scatter(distances, actual_times, alpha=0.6, label='Actual Data', color='#2ecc71', s=50)
    
    sorted_idx = np.argsort(distances.flatten())
    ax1.plot(distances[sorted_idx], predicted_times[sorted_idx], 
             color='#9b59b6', linewidth=2, label='Sigmoid NN Predictions')
    
    ax1.set_xlabel('Distance (km)', fontsize=12)
    ax1.set_ylabel('Delivery Time (minutes)', fontsize=12)
    ax1.set_title('Delivery Time Prediction (Sigmoid)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss
    ax2 = axes[1]
    ax2.plot(losses, color='#9b59b6', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.set_title('Training Loss Over Time (Sigmoid)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('delivery_sigmoid_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_activations():
    """
    Compare Sigmoid vs ReLU activation functions.
    """
    print("\n" + "=" * 60)
    print("   Sigmoid vs ReLU Comparison")
    print("=" * 60)
    print()
    
    comparison = """
    | Property              | Sigmoid                  | ReLU                    |
    |-----------------------|--------------------------|-------------------------|
    | Formula               | 1/(1+e^(-x))            | max(0, x)               |
    | Output Range          | (0, 1)                   | [0, infinity)           |
    | Derivative            | sigma(1-sigma)           | 0 or 1                  |
    | Max Gradient          | 0.25                     | 1                       |
    | Vanishing Gradient    | Yes (for large |x|)      | No (for x > 0)          |
    | Dead Neurons          | No                       | Yes (for x < 0)         |
    | Computation           | Expensive (exp)          | Cheap (comparison)      |
    | Zero-Centered         | No (output > 0)          | No (output >= 0)        |
    | Common Use            | Binary classification    | Hidden layers in DNNs   |
    """
    print(comparison)


def main():
    print("=" * 60)
    print("   Neural Network with SIGMOID - Delivery Time Prediction")
    print("=" * 60)
    print()
    
    # Show sigmoid function visualization
    if HAS_MATPLOTLIB:
        print("Visualizing Sigmoid function...")
        plot_sigmoid_function()
        print("   Saved plot to 'sigmoid_function.png'")
        print()
    
    # Generate synthetic data
    print("Generating delivery data...")
    distances, delivery_times = generate_delivery_data(n_samples=200, noise=0.15)
    
    # Normalize the data
    X_norm, X_min, X_max = normalize(distances)
    y_norm, y_min, y_max = normalize(delivery_times)
    
    # Split into training and test sets (80/20)
    split_idx = int(0.8 * len(distances))
    X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
    y_train, y_test = y_norm[:split_idx], y_norm[split_idx:]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print()
    
    # Create and train the neural network with Sigmoid
    print("Creating Neural Network with SIGMOID activation...")
    print("   Architecture: 1 -> 8 -> 1 (input -> hidden -> output)")
    print("   Activation: Sigmoid (hidden layer)")
    print()
    
    nn = NeuralNetworkSigmoid(
        input_size=1,
        hidden_size=8,
        output_size=1,
        learning_rate=0.5  # Higher learning rate for sigmoid
    )
    
    print("Training the network...")
    print("-" * 40)
    losses = nn.train(X_train, y_train, epochs=1000, verbose=True)
    print("-" * 40)
    print()
    
    # Evaluate on test set
    y_test_pred = nn.predict(X_test)
    test_loss = nn.compute_loss(y_test_pred, y_test)
    print(f"Test Loss (MSE): {test_loss:.6f}")
    
    # Denormalize for visualization
    y_pred_all = nn.predict(X_norm)
    y_pred_denorm = denormalize(y_pred_all, y_min, y_max)
    
    # Calculate metrics
    mae = np.mean(np.abs(y_pred_denorm - delivery_times))
    print(f"Mean Absolute Error: {mae:.2f} minutes")
    print()
    
    # Interactive predictions
    print("=" * 60)
    print("   Make Predictions")
    print("=" * 60)
    
    test_distances = [5, 10, 20, 30, 45]
    print("\nPredicted delivery times:")
    print("-" * 40)
    print(f"{'Distance (km)':<15} {'Predicted Time (min)':<20}")
    print("-" * 40)
    
    for dist in test_distances:
        dist_norm = (dist - X_min) / (X_max - X_min)
        dist_input = np.array([[dist_norm]])
        pred_norm = nn.predict(dist_input)
        pred_time = denormalize(pred_norm, y_min, y_max)[0, 0]
        print(f"{dist:<15} {pred_time:<20.1f}")
    
    print("-" * 40)
    print()
    
    # Compare activations
    compare_activations()
    
    # Plot results
    if HAS_MATPLOTLIB:
        print("\nGenerating visualization...")
        plot_results(distances, delivery_times, y_pred_denorm, losses)
        print("   Saved plot to 'delivery_sigmoid_results.png'")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()

