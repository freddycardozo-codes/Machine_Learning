"""
Neural Network from Scratch - Delivery Time Prediction
Predicts delivery time based on distance without using PyTorch
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    A simple feedforward neural network with one hidden layer.
    Architecture: Input(1) -> Hidden(10) -> Output(1)
    """
    
    def __init__(self, input_size=1, hidden_size=10, output_size=1, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Store for backpropagation
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def forward(self, X):
        """Forward pass through the network"""
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Output layer (linear activation for regression)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Linear activation
        
        return self.a2
    
    def backward(self, X, y):
        """Backward pass - compute gradients and update weights"""
        m = X.shape[0]  # Number of samples
        
        # Output layer gradients
        dz2 = self.a2 - y  # Derivative of MSE loss
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def compute_loss(self, y_pred, y_true):
        """Mean Squared Error loss"""
        return np.mean((y_pred - y_true) ** 2)
    
    def train(self, X, y, epochs=1000, verbose=True):
        """Train the neural network"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)


def generate_delivery_data(n_samples=100, noise=0.1):
    """
    Generate synthetic delivery data.
    Assumes delivery time increases with distance (with some non-linearity)
    
    Formula: time = 10 + 2*distance + 0.01*distance^2 + noise
    (Base time 10 min + 2 min per km + slight increase for longer distances)
    """
    np.random.seed(42)
    
    # Distance in kilometers (1 to 50 km)
    distances = np.random.uniform(1, 50, n_samples).reshape(-1, 1)
    
    # Delivery time in minutes
    times = 10 + 2 * distances + 0.01 * (distances ** 2) + np.random.normal(0, noise * 10, (n_samples, 1))
    
    return distances, times


def normalize(data):
    """Normalize data to [0, 1] range"""
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val), min_val, max_val


def denormalize(data, min_val, max_val):
    """Denormalize data back to original range"""
    return data * (max_val - min_val) + min_val


def main():
    print("=" * 60)
    print("Neural Network from Scratch - Delivery Time Prediction")
    print("=" * 60)
    
    # Generate training data
    print("\n[1] Generating synthetic delivery data...")
    distances, times = generate_delivery_data(n_samples=200)
    print(f"    Generated {len(distances)} samples")
    print(f"    Distance range: {distances.min():.1f} - {distances.max():.1f} km")
    print(f"    Time range: {times.min():.1f} - {times.max():.1f} minutes")
    
    # Normalize data for better training
    X_norm, X_min, X_max = normalize(distances)
    y_norm, y_min, y_max = normalize(times)
    
    # Split into training and testing sets (80/20)
    split_idx = int(0.8 * len(distances))
    X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
    y_train, y_test = y_norm[:split_idx], y_norm[split_idx:]
    
    print(f"    Training samples: {len(X_train)}")
    print(f"    Testing samples: {len(X_test)}")
    
    # Create and train the neural network
    print("\n[2] Training Neural Network...")
    print("    Architecture: Input(1) -> Hidden(10, ReLU) -> Output(1)")
    
    nn = NeuralNetwork(input_size=1, hidden_size=10, output_size=1, learning_rate=0.1)
    losses = nn.train(X_train, y_train, epochs=1000, verbose=True)
    
    # Evaluate on test set
    print("\n[3] Evaluating on test set...")
    y_pred_norm = nn.predict(X_test)
    y_pred = denormalize(y_pred_norm, y_min, y_max)
    y_actual = denormalize(y_test, y_min, y_max)
    
    # Calculate metrics
    mse = np.mean((y_pred - y_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_actual))
    
    print(f"    Mean Squared Error (MSE): {mse:.4f}")
    print(f"    Root Mean Squared Error (RMSE): {rmse:.4f} minutes")
    print(f"    Mean Absolute Error (MAE): {mae:.4f} minutes")
    
    # Interactive predictions
    print("\n[4] Sample Predictions:")
    print("-" * 40)
    test_distances = [5, 10, 15, 25, 35, 45]
    
    for dist in test_distances:
        # Normalize input
        dist_norm = (dist - X_min) / (X_max - X_min)
        
        # Predict
        pred_norm = nn.predict(np.array([[dist_norm]]))
        pred_time = denormalize(pred_norm, y_min, y_max)[0, 0]
        
        # Expected time (using our formula)
        expected = 10 + 2 * dist + 0.01 * (dist ** 2)
        
        print(f"    Distance: {dist:2d} km -> Predicted: {pred_time:.1f} min (Expected: ~{expected:.1f} min)")
    
    # Plot results
    print("\n[5] Generating visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Training Loss
    axes[0].plot(losses, color='#E74C3C', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor('#F8F9FA')
    
    # Plot 2: Predictions vs Actual
    X_test_denorm = denormalize(X_test, X_min, X_max)
    axes[1].scatter(X_test_denorm, y_actual, alpha=0.7, c='#3498DB', label='Actual', s=60)
    axes[1].scatter(X_test_denorm, y_pred, alpha=0.7, c='#E74C3C', marker='x', label='Predicted', s=60)
    axes[1].set_xlabel('Distance (km)', fontsize=12)
    axes[1].set_ylabel('Delivery Time (min)', fontsize=12)
    axes[1].set_title('Predictions vs Actual (Test Set)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor('#F8F9FA')
    
    # Plot 3: Full prediction curve
    X_full = np.linspace(0, 1, 100).reshape(-1, 1)
    y_full_pred = nn.predict(X_full)
    X_full_denorm = denormalize(X_full, X_min, X_max)
    y_full_denorm = denormalize(y_full_pred, y_min, y_max)
    
    axes[2].scatter(distances, times, alpha=0.5, c='#3498DB', label='Training Data', s=40)
    axes[2].plot(X_full_denorm, y_full_denorm, color='#E74C3C', linewidth=3, label='NN Prediction')
    axes[2].set_xlabel('Distance (km)', fontsize=12)
    axes[2].set_ylabel('Delivery Time (min)', fontsize=12)
    axes[2].set_title('Neural Network Fitted Curve', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_facecolor('#F8F9FA')
    
    plt.tight_layout()
    plt.savefig('delivery_nn_results.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("    Saved visualization to 'delivery_nn_results.png'")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


