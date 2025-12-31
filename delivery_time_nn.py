"""
Neural Network from Scratch - Delivery Time Prediction
========================================================
A simple neural network implementation without PyTorch to predict
delivery time based on distance traveled.

This script demonstrates:
- Building a neural network from scratch using only NumPy
- Forward propagation (computing predictions)
- Backpropagation (computing gradients for learning)
- Gradient descent optimization
- Data normalization techniques
- Model evaluation and visualization

Author: Neural Network Tutorial
"""

# Import NumPy - the fundamental library for numerical computing in Python
# NumPy provides efficient array operations and mathematical functions
import numpy as np

# Try to import matplotlib for visualization (optional)
# We wrap this in try/except to make the script work even without matplotlib
try:
    # matplotlib.pyplot provides MATLAB-like plotting interface
    import matplotlib.pyplot as plt
    # Set flag to True indicating matplotlib is available
    HAS_MATPLOTLIB = True
except ImportError:
    # If matplotlib is not installed, set flag to False
    HAS_MATPLOTLIB = False
    # Inform the user that visualization will be skipped
    print("Note: matplotlib not available. Visualization will be skipped.")


class NeuralNetwork:
    """
    A simple feedforward neural network with one hidden layer.
    
    Architecture:
        Input Layer:  1 neuron (distance)
        Hidden Layer: 8 neurons (ReLU activation)
        Output Layer: 1 neuron (delivery time)
    
    This is a regression network that predicts continuous values (delivery time)
    based on input features (distance).
    """
    
    def __init__(self, input_size=1, hidden_size=8, output_size=1, learning_rate=0.001):
        """
        Initialize the neural network with random weights.
        
        Args:
            input_size: Number of input features (default 1 for distance)
            hidden_size: Number of neurons in hidden layer (default 8)
            output_size: Number of output neurons (default 1 for delivery time)
            learning_rate: Step size for gradient descent (default 0.001)
        """
        # Store the learning rate as an instance variable
        # Learning rate controls how much we adjust weights during training
        # Too high = unstable training, too low = slow convergence
        self.learning_rate = learning_rate
        
        # Initialize weights for layer 1 (input -> hidden) using He initialization
        # Xavier/He initialization helps prevent vanishing/exploding gradients
        # np.random.randn generates random numbers from standard normal distribution
        # Multiplying by sqrt(2/n) scales weights appropriately for ReLU activation
        # Shape: (input_size, hidden_size) = (1, 8) for our case
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        
        # Initialize biases for layer 1 as zeros
        # Shape: (1, hidden_size) = (1, 8) for broadcasting with batch inputs
        # Biases allow neurons to have non-zero output even when inputs are zero
        self.b1 = np.zeros((1, hidden_size))
        
        # Initialize weights for layer 2 (hidden -> output) using He initialization
        # Shape: (hidden_size, output_size) = (8, 1) for our case
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        
        # Initialize biases for layer 2 as zeros
        # Shape: (1, output_size) = (1, 1) for our case
        self.b2 = np.zeros((1, output_size))
        
        # Initialize placeholders for intermediate values needed in backpropagation
        # z1: Pre-activation values of hidden layer (before ReLU)
        self.z1 = None
        # a1: Activation values of hidden layer (after ReLU)
        self.a1 = None
        # z2: Pre-activation values of output layer
        self.z2 = None
        # a2: Activation values of output layer (final prediction)
        self.a2 = None
    
    def relu(self, x):
        """
        ReLU (Rectified Linear Unit) activation function.
        
        Formula: f(x) = max(0, x)
        - Returns x if x > 0, otherwise returns 0
        - Introduces non-linearity to the network
        - Helps with vanishing gradient problem compared to sigmoid/tanh
        
        Args:
            x: Input array of any shape
            
        Returns:
            Array of same shape with negative values replaced by 0
        """
        # np.maximum compares element-wise and returns the larger value
        # This clips all negative values to 0, keeping positive values unchanged
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """
        Derivative of ReLU activation function.
        
        Formula: f'(x) = 1 if x > 0, else 0
        - Used during backpropagation to compute gradients
        - Gradient is 1 for positive inputs (passes gradient through)
        - Gradient is 0 for negative inputs (blocks gradient flow)
        
        Args:
            x: Input array (pre-activation values z1)
            
        Returns:
            Array of same shape with 1s where x > 0 and 0s elsewhere
        """
        # (x > 0) creates boolean array: True where x > 0, False elsewhere
        # .astype(float) converts True -> 1.0 and False -> 0.0
        return (x > 0).astype(float)
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        This computes the network's prediction by passing input through each layer:
        Input -> Hidden Layer (with ReLU) -> Output Layer (linear)
        
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
        
        # Apply ReLU activation function to introduce non-linearity
        # Without activation, the entire network would be just a linear function
        # a1 has shape (n_samples, 8) - one activation per hidden neuron
        self.a1 = self.relu(self.z1)
        
        # OUTPUT LAYER COMPUTATION
        # Compute the weighted sum for output layer
        # z2 = a1 * W2 + b2
        # Matrix multiplication: (n_samples, 8) @ (8, 1) = (n_samples, 1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        
        # Use linear activation (identity function) for output layer
        # For regression problems, we want unbounded continuous output
        # a2 = z2 (no transformation applied)
        self.a2 = self.z2  # Linear activation
        
        # Return the final prediction (delivery time estimate)
        return self.a2
    
    def backward(self, X, y):
        """
        Backpropagation to compute gradients and update weights.
        
        This implements the chain rule to compute how much each weight
        contributed to the prediction error, then adjusts weights accordingly.
        
        The gradient flow: Loss -> Output -> Hidden -> Input
        
        Args:
            X: Input features of shape (n_samples, input_size)
            y: True labels (delivery times) of shape (n_samples, output_size)
        """
        # Get the number of training samples for averaging gradients
        # We divide by m to get the mean gradient across all samples
        m = X.shape[0]  # Number of samples
        
        # ==================== OUTPUT LAYER GRADIENTS ====================
        
        # Compute error at output layer (derivative of MSE loss with respect to predictions)
        # MSE Loss = (1/2m) * sum((y_pred - y_true)^2)
        # d(Loss)/d(a2) = (1/m) * (y_pred - y_true) = (1/m) * (a2 - y)
        # For simplicity, we compute (a2 - y) and scale in the weight updates
        # dz2 has shape (n_samples, 1)
        dz2 = self.a2 - y  # Derivative of MSE loss
        
        # Compute gradient of loss with respect to W2
        # d(Loss)/d(W2) = d(Loss)/d(z2) * d(z2)/d(W2)
        # Since z2 = a1 * W2 + b2, d(z2)/d(W2) = a1^T
        # np.dot(a1.T, dz2): (8, n_samples) @ (n_samples, 1) = (8, 1)
        # Divide by m to get average gradient
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        
        # Compute gradient of loss with respect to b2
        # d(Loss)/d(b2) = d(Loss)/d(z2) * d(z2)/d(b2) = dz2 * 1
        # Sum across all samples and divide by m for average
        # keepdims=True maintains the shape (1, 1) for broadcasting
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # ==================== HIDDEN LAYER GRADIENTS ====================
        
        # Propagate error back to hidden layer
        # d(Loss)/d(a1) = d(Loss)/d(z2) * d(z2)/d(a1)
        # Since z2 = a1 * W2 + b2, d(z2)/d(a1) = W2^T
        # np.dot(dz2, W2.T): (n_samples, 1) @ (1, 8) = (n_samples, 8)
        da1 = np.dot(dz2, self.W2.T)
        
        # Apply chain rule through ReLU activation
        # d(Loss)/d(z1) = d(Loss)/d(a1) * d(a1)/d(z1)
        # d(a1)/d(z1) is the ReLU derivative: 1 if z1 > 0, else 0
        # Element-wise multiplication preserves shape (n_samples, 8)
        dz1 = da1 * self.relu_derivative(self.z1)
        
        # Compute gradient of loss with respect to W1
        # d(Loss)/d(W1) = d(Loss)/d(z1) * d(z1)/d(W1)
        # Since z1 = X * W1 + b1, d(z1)/d(W1) = X^T
        # np.dot(X.T, dz1): (1, n_samples) @ (n_samples, 8) = (1, 8)
        dW1 = (1/m) * np.dot(X.T, dz1)
        
        # Compute gradient of loss with respect to b1
        # Sum across all samples, divide by m, keep dimensions
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # ==================== WEIGHT UPDATES (GRADIENT DESCENT) ====================
        
        # Update weights and biases using gradient descent rule:
        # new_weight = old_weight - learning_rate * gradient
        # The negative sign means we move in the direction that decreases loss
        
        # Update output layer weights
        # W2 shape: (8, 1), dW2 shape: (8, 1)
        self.W2 -= self.learning_rate * dW2
        
        # Update output layer biases
        # b2 shape: (1, 1), db2 shape: (1, 1)
        self.b2 -= self.learning_rate * db2
        
        # Update hidden layer weights
        # W1 shape: (1, 8), dW1 shape: (1, 8)
        self.W1 -= self.learning_rate * dW1
        
        # Update hidden layer biases
        # b1 shape: (1, 8), db1 shape: (1, 8)
        self.b1 -= self.learning_rate * db1
    
    def compute_loss(self, y_pred, y_true):
        """
        Compute Mean Squared Error (MSE) loss.
        
        MSE measures the average squared difference between predictions and actual values.
        Formula: MSE = (1/n) * sum((y_pred - y_true)^2)
        
        Why MSE?
        - Penalizes larger errors more heavily (due to squaring)
        - Always positive (no cancellation between positive and negative errors)
        - Differentiable everywhere (smooth optimization landscape)
        
        Args:
            y_pred: Predicted values from the network
            y_true: Actual target values
            
        Returns:
            Scalar MSE loss value
        """
        # (y_pred - y_true) computes element-wise difference (residuals)
        # ** 2 squares each difference (makes all values positive, penalizes large errors)
        # np.mean() averages across all samples to get a single loss value
        return np.mean((y_pred - y_true) ** 2)
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network using batch gradient descent.
        
        Training loop:
        1. Forward pass: Compute predictions
        2. Compute loss: Measure prediction error
        3. Backward pass: Compute gradients and update weights
        4. Repeat for specified number of epochs
        
        Args:
            X: Training features of shape (n_samples, input_size)
            y: Training labels of shape (n_samples, output_size)
            epochs: Number of complete passes through the training data (default 1000)
            verbose: If True, print progress every 100 epochs (default True)
            
        Returns:
            List of loss values at each epoch (for plotting learning curve)
        """
        # Initialize empty list to store loss at each epoch
        # This allows us to visualize the training progress
        losses = []
        
        # Training loop - iterate for the specified number of epochs
        # An epoch is one complete pass through all training data
        for epoch in range(epochs):
            # ===== STEP 1: FORWARD PASS =====
            # Compute predictions for all training samples
            # y_pred shape: (n_samples, 1)
            y_pred = self.forward(X)
            
            # ===== STEP 2: COMPUTE LOSS =====
            # Calculate how far off our predictions are from actual values
            # Lower loss = better predictions
            loss = self.compute_loss(y_pred, y)
            
            # Store the loss for this epoch in our history list
            # Used later for visualizing training progress
            losses.append(loss)
            
            # ===== STEP 3: BACKWARD PASS =====
            # Compute gradients and update all weights and biases
            # This is where the network actually learns
            self.backward(X, y)
            
            # ===== STEP 4: PRINT PROGRESS =====
            # Print loss every 100 epochs if verbose mode is enabled
            # (epoch + 1) because epoch starts at 0, but we want to show 100, 200, etc.
            # (epoch + 1) % 100 == 0 is True when epoch+1 is divisible by 100
            if verbose and (epoch + 1) % 100 == 0:
                # f-string for formatted output
                # {epoch + 1} shows current epoch number
                # {epochs} shows total epochs
                # {loss:.4f} shows loss with 4 decimal places
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        # Return the complete loss history for analysis and visualization
        return losses
    
    def predict(self, X):
        """
        Make predictions for new data.
        
        This is simply a wrapper around forward() for clarity.
        Use this method when making predictions on new/test data.
        
        Args:
            X: Input features (should be normalized the same way as training data)
            
        Returns:
            Predicted values (will need denormalization for actual delivery times)
        """
        # Call forward propagation to compute predictions
        # No learning happens here - just inference
        return self.forward(X)


def generate_delivery_data(n_samples=100, noise=0.1):
    """
    Generate synthetic delivery data for training.
    
    This simulates a realistic delivery scenario where:
    - There's a base preparation time (constant for all orders)
    - Travel time increases linearly with distance
    - Random factors (traffic, weather, etc.) add noise
    
    The relationship is approximately:
    delivery_time = base_time + time_per_km * distance + noise
    
    Args:
        n_samples: Number of data points to generate (default 100)
        noise: Amount of random variation as a fraction (default 0.1 = 10%)
        
    Returns:
        Tuple of (distances, delivery_times) as numpy arrays
        Both have shape (n_samples, 1)
    """
    # Set random seed for reproducibility
    # Using seed 42 ensures we get the same "random" data every time
    # This is important for debugging and comparing results
    np.random.seed(42)
    
    # Generate random distances between 1 and 50 kilometers
    # np.random.uniform(low, high, size) generates uniform random numbers
    # .reshape(-1, 1) converts from shape (n_samples,) to (n_samples, 1)
    # -1 means "infer this dimension" = n_samples
    # This 2D shape is needed for matrix operations in the neural network
    distances = np.random.uniform(1, 50, n_samples).reshape(-1, 1)
    
    # Define the delivery time formula parameters
    # base_time: Fixed time for order preparation, packaging, etc.
    base_time = 10  # 10 minutes base preparation time
    
    # time_per_km: Average travel time per kilometer
    # This represents the core linear relationship we want the network to learn
    time_per_km = 2  # 2 minutes per kilometer
    
    # Calculate delivery times using the formula with noise
    delivery_times = (
        # Base preparation time (constant added to all)
        base_time + 
        # Linear travel time component (main relationship)
        time_per_km * distances + 
        # Random noise that scales with distance (longer trips = more variability)
        # np.random.normal(mean, std, shape) generates Gaussian noise
        # mean=0: centered around zero (no systematic bias)
        # std=noise*distances: standard deviation proportional to distance
        # This creates heteroscedasticity (variance increases with distance)
        np.random.normal(0, noise * distances, (n_samples, 1))
    )
    
    # Return both arrays as a tuple
    return distances, delivery_times


def normalize(data, min_val=None, max_val=None):
    """
    Min-max normalization to scale data to [0, 1] range.
    
    Formula: x_normalized = (x - min) / (max - min)
    
    Why normalize?
    - Neural networks train better when input features are on similar scales
    - Prevents features with large values from dominating
    - Helps gradient descent converge faster
    - Avoids numerical overflow/underflow issues
    
    Args:
        data: Input array to normalize
        min_val: Minimum value to use (if None, computed from data)
        max_val: Maximum value to use (if None, computed from data)
        
    Returns:
        Tuple of (normalized_data, min_val, max_val)
        min_val and max_val are returned for denormalization later
    """
    # If min_val not provided, compute it from the data
    # data.min() finds the smallest value in the array
    if min_val is None:
        min_val = data.min()
    
    # If max_val not provided, compute it from the data
    # data.max() finds the largest value in the array
    if max_val is None:
        max_val = data.max()
    
    # Apply min-max normalization formula
    # (data - min_val) shifts data so minimum becomes 0
    # Dividing by (max_val - min_val) scales so maximum becomes 1
    # All values end up in the range [0, 1]
    # Return the normalized data along with min/max for later denormalization
    return (data - min_val) / (max_val - min_val), min_val, max_val


def denormalize(data, min_val, max_val):
    """
    Reverse min-max normalization to recover original scale.
    
    Formula: x_original = x_normalized * (max - min) + min
    
    This is the inverse of the normalize function.
    Used to convert predictions back to actual delivery time values.
    
    Args:
        data: Normalized data in [0, 1] range
        min_val: Original minimum value used for normalization
        max_val: Original maximum value used for normalization
        
    Returns:
        Data restored to original scale
    """
    # Reverse the normalization formula:
    # 1. Multiply by range (max - min) to restore the scale
    # 2. Add min to shift back to original position
    return data * (max_val - min_val) + min_val


def plot_results(distances, actual_times, predicted_times, losses):
    """
    Visualize the training results with two plots.
    
    Plot 1: Predictions vs Actual data (shows model fit)
    Plot 2: Training loss over epochs (shows learning progress)
    
    Args:
        distances: Original distance values (x-axis for plot 1)
        actual_times: Ground truth delivery times (scatter points)
        predicted_times: Model predictions (line plot)
        losses: List of loss values per epoch (plot 2)
    """
    # Check if matplotlib is available before attempting to plot
    # If not available, print a message and return early
    if not HAS_MATPLOTLIB:
        print("   (Skipping plot - matplotlib not available)")
        return  # Exit the function without creating plots
    
    # Create a figure with 1 row and 2 columns of subplots
    # figsize=(14, 5) sets the figure size to 14 inches wide, 5 inches tall
    # Returns: fig (the figure object) and axes (array of 2 subplot axes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ==================== PLOT 1: PREDICTIONS VS ACTUAL ====================
    
    # Select the first (left) subplot
    ax1 = axes[0]
    
    # Create a scatter plot of actual data points
    # distances: x-coordinates (distance in km)
    # actual_times: y-coordinates (actual delivery time)
    # alpha=0.6: 60% opacity (allows seeing overlapping points)
    # label: text for legend
    # color: '#2ecc71' is a green hex color
    # s=50: marker size of 50 points
    ax1.scatter(distances, actual_times, alpha=0.6, label='Actual Data', color='#2ecc71', s=50)
    
    # Sort data by distance for a smooth prediction line
    # np.argsort returns indices that would sort the array
    # .flatten() converts 2D array to 1D for indexing
    sorted_idx = np.argsort(distances.flatten())
    
    # Plot the neural network predictions as a line
    # Using sorted indices ensures the line is drawn in order (no zigzag)
    # distances[sorted_idx]: sorted x values
    # predicted_times[sorted_idx]: corresponding y predictions
    # color='#e74c3c': red hex color
    # linewidth=2: line thickness
    ax1.plot(distances[sorted_idx], predicted_times[sorted_idx], 
             color='#e74c3c', linewidth=2, label='NN Predictions')
    
    # Set x-axis label with font size 12
    ax1.set_xlabel('Distance (km)', fontsize=12)
    
    # Set y-axis label with font size 12
    ax1.set_ylabel('Delivery Time (minutes)', fontsize=12)
    
    # Set plot title with larger font and bold weight
    ax1.set_title('Delivery Time Prediction', fontsize=14, fontweight='bold')
    
    # Add legend to identify scatter vs line
    ax1.legend()
    
    # Add grid lines with 30% opacity for readability
    ax1.grid(True, alpha=0.3)
    
    # ==================== PLOT 2: TRAINING LOSS ====================
    
    # Select the second (right) subplot
    ax2 = axes[1]
    
    # Plot the loss values over epochs
    # losses is a list where index = epoch number
    # color='#3498db': blue hex color
    # linewidth=1.5: line thickness
    ax2.plot(losses, color='#3498db', linewidth=1.5)
    
    # Set x-axis label (epoch number)
    ax2.set_xlabel('Epoch', fontsize=12)
    
    # Set y-axis label (loss value)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    
    # Set plot title
    ax2.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    
    # Add grid lines
    ax2.grid(True, alpha=0.3)
    
    # Use logarithmic scale for y-axis
    # This better shows the rapid initial decrease followed by slow convergence
    ax2.set_yscale('log')
    
    # ==================== SAVE AND DISPLAY ====================
    
    # Adjust subplot layout to prevent overlapping
    plt.tight_layout()
    
    # Save the figure to a file
    # dpi=150: resolution of 150 dots per inch
    # bbox_inches='tight': crop whitespace from edges
    plt.savefig('delivery_nn_results.png', dpi=150, bbox_inches='tight')
    
    # Display the plot in a window (blocks until window is closed)
    plt.show()


def main():
    """
    Main function that orchestrates the entire training and evaluation pipeline.
    
    Steps:
    1. Generate synthetic delivery data
    2. Normalize the data
    3. Split into training and test sets
    4. Create and train the neural network
    5. Evaluate on test set
    6. Make sample predictions
    7. Visualize results
    """
    # Print a decorative header for the program output
    # "=" * 60 creates a string of 60 equal signs
    print("=" * 60)
    print("   Neural Network from Scratch - Delivery Time Prediction")
    print("=" * 60)
    # Print empty line for spacing
    print()
    
    # ==================== DATA GENERATION ====================
    
    # Generate synthetic training data
    print("Generating delivery data...")
    
    # Call the data generation function
    # n_samples=200: create 200 data points
    # noise=0.15: 15% noise level for realistic variation
    # Returns two arrays: distances (input X) and delivery_times (target y)
    distances, delivery_times = generate_delivery_data(n_samples=200, noise=0.15)
    
    # ==================== DATA PREPROCESSING ====================
    
    # Normalize input features (distances) to [0, 1] range
    # X_norm: normalized distances
    # X_min, X_max: original min/max values needed for denormalization
    X_norm, X_min, X_max = normalize(distances)
    
    # Normalize target values (delivery times) to [0, 1] range
    # y_norm: normalized delivery times
    # y_min, y_max: original min/max values needed for denormalization
    y_norm, y_min, y_max = normalize(delivery_times)
    
    # ==================== TRAIN/TEST SPLIT ====================
    
    # Calculate the index to split data at 80% for training
    # int() truncates to integer (e.g., 160 for 200 samples)
    split_idx = int(0.8 * len(distances))
    
    # Split features into training (first 80%) and test (last 20%) sets
    # X_train: samples from index 0 to split_idx-1
    # X_test: samples from split_idx to end
    X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
    
    # Split targets similarly
    # y_train: targets for training samples
    # y_test: targets for test samples (held out for evaluation)
    y_train, y_test = y_norm[:split_idx], y_norm[split_idx:]
    
    # Print dataset sizes for confirmation
    # len() returns the number of samples
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print()
    
    # ==================== MODEL CREATION ====================
    
    print("Creating Neural Network...")
    
    # Print the architecture for user understanding
    print("   Architecture: 1 -> 8 -> 1 (input -> hidden -> output)")
    print()
    
    # Instantiate the neural network with specified hyperparameters
    # input_size=1: one input feature (distance)
    # hidden_size=8: 8 neurons in hidden layer (can be tuned)
    # output_size=1: one output (delivery time)
    # learning_rate=0.1: relatively high for faster convergence on simple data
    nn = NeuralNetwork(
        input_size=1,
        hidden_size=8,
        output_size=1,
        learning_rate=0.1
    )
    
    # ==================== MODEL TRAINING ====================
    
    print("Training the network...")
    # Print separator for visual clarity
    print("-" * 40)
    
    # Train the neural network
    # X_train, y_train: training data
    # epochs=1000: 1000 passes through the training data
    # verbose=True: print progress every 100 epochs
    # Returns list of loss values at each epoch
    losses = nn.train(X_train, y_train, epochs=1000, verbose=True)
    
    # Print closing separator
    print("-" * 40)
    print()
    
    # ==================== MODEL EVALUATION ====================
    
    # Make predictions on the test set (data the model hasn't seen during training)
    # This evaluates generalization ability
    y_test_pred = nn.predict(X_test)
    
    # Compute MSE loss on test set
    # Lower test loss indicates better generalization
    test_loss = nn.compute_loss(y_test_pred, y_test)
    
    # Print test loss with 6 decimal places
    print(f"Test Loss (MSE): {test_loss:.6f}")
    
    # Make predictions on ALL data for visualization
    # This includes both training and test samples
    y_pred_all = nn.predict(X_norm)
    
    # Denormalize predictions back to original scale (minutes)
    # This converts from [0,1] range back to actual delivery times
    y_pred_denorm = denormalize(y_pred_all, y_min, y_max)
    
    # Calculate Mean Absolute Error (MAE) as an interpretable metric
    # MAE = average of |prediction - actual| across all samples
    # np.abs computes absolute values
    # np.mean averages across all samples
    # This tells us "on average, predictions are off by X minutes"
    mae = np.mean(np.abs(y_pred_denorm - delivery_times))
    
    # Print MAE with 2 decimal places
    print(f"Mean Absolute Error: {mae:.2f} minutes")
    print()
    
    # ==================== INTERACTIVE PREDICTIONS ====================
    
    # Print section header
    print("=" * 60)
    print("   Make Predictions")
    print("=" * 60)
    
    # Define test distances for demonstration
    # These are example distances we want to predict delivery times for
    test_distances = [5, 10, 20, 30, 45]
    
    # Print table header
    print("\nPredicted delivery times:")
    print("-" * 40)
    # Print column headers with padding
    # {:<15} means left-align in 15-character width
    # {:<20} means left-align in 20-character width
    print(f"{'Distance (km)':<15} {'Predicted Time (min)':<20}")
    print("-" * 40)
    
    # Loop through each test distance and make predictions
    for dist in test_distances:
        # Normalize the input distance using training data statistics
        # This is crucial: we must use the same min/max as training data
        # Formula: (x - min) / (max - min)
        dist_norm = (dist - X_min) / (X_max - X_min)
        
        # Reshape to 2D array for the network: shape (1, 1)
        # [[dist_norm]] creates a 2D array with 1 row and 1 column
        dist_input = np.array([[dist_norm]])
        
        # Make prediction using the trained network
        # pred_norm is in normalized [0,1] scale
        pred_norm = nn.predict(dist_input)
        
        # Denormalize prediction to actual delivery time in minutes
        # [0, 0] extracts the scalar value from 2D array
        pred_time = denormalize(pred_norm, y_min, y_max)[0, 0]
        
        # Print the distance and predicted time in formatted columns
        # {dist:<15}: distance left-aligned in 15 chars
        # {pred_time:<20.1f}: time left-aligned in 20 chars with 1 decimal
        print(f"{dist:<15} {pred_time:<20.1f}")
    
    # Print closing separator
    print("-" * 40)
    print()
    
    # ==================== VISUALIZATION ====================
    
    # Check if matplotlib is available before generating plots
    if HAS_MATPLOTLIB:
        print("Generating visualization...")
        
        # Call the plotting function with all required data
        # distances: original x values
        # delivery_times: actual y values
        # y_pred_denorm: model predictions (denormalized)
        # losses: training loss history
        plot_results(distances, delivery_times, y_pred_denorm, losses)
        
        # Inform user where the plot was saved
        print("   Saved plot to 'delivery_nn_results.png'")
    
    # Final empty line for spacing
    print()
    
    # Print completion message
    print("Done!")


# This is the standard Python idiom for making a script executable
# __name__ is a special variable that equals "__main__" when the script is run directly
# (as opposed to being imported as a module)
# This ensures main() only runs when executing this file directly
if __name__ == "__main__":
    # Call the main function to start the program
    main()
