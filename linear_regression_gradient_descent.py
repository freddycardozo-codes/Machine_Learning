"""
Linear Regression with Gradient Descent from Scratch
=====================================================
A comprehensive implementation demonstrating the concepts from:
https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent

This program illustrates:
1. Linear regression model: y = wx + b
2. Loss function: Mean Squared Error (MSE)
3. Gradient descent optimization algorithm
4. Model convergence visualization
5. Convex loss surface (3D visualization)
6. Training snapshots at different iterations

Key Concepts:
- Gradient descent iteratively finds weights and bias that minimize loss
- The model converges when additional iterations don't reduce loss significantly
- Linear models have convex loss surfaces, guaranteeing finding the global minimum

Author: Neural Network Tutorial
"""

# ==================== IMPORTS ====================

# NumPy for numerical computations
# Provides efficient array operations for gradient calculations
import numpy as np

# Try to import matplotlib for visualization
try:
    # matplotlib.pyplot for 2D plotting (loss curves, model fit)
    import matplotlib.pyplot as plt
    # mplot3d enables 3D plotting capabilities for loss surface
    from mpl_toolkits.mplot3d import Axes3D
    # Set flag indicating matplotlib is available
    HAS_MATPLOTLIB = True
except ImportError:
    # If matplotlib not installed, skip visualizations
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available. Visualization will be skipped.")


class LinearRegressionGD:
    """
    Linear Regression model trained using Gradient Descent.
    
    Model: y = w * x + b
    
    Where:
    - y: predicted output (dependent variable)
    - x: input feature (independent variable)
    - w: weight (slope of the line)
    - b: bias (y-intercept)
    
    The goal is to find w and b that minimize the loss function.
    """
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize the Linear Regression model.
        
        Args:
            learning_rate: Step size for gradient descent updates.
                          Controls how much to adjust w and b each iteration.
                          - Too large: may overshoot minimum
                          - Too small: slow convergence
        """
        # Store the learning rate (also called step size or alpha)
        # This hyperparameter controls the magnitude of weight updates
        self.learning_rate = learning_rate
        
        # Initialize weight (w) to a small random value near zero
        # Starting near zero is a common practice
        # np.random.randn() returns sample from standard normal distribution
        # Multiply by 0.01 to get small initial value
        self.w = np.random.randn() * 0.01
        
        # Initialize bias (b) to zero
        # Bias can start at zero without issues
        self.b = 0.0
        
        # Lists to store training history for visualization
        # These track how w, b, and loss change during training
        self.history = {
            'weights': [],      # Weight values at each iteration
            'biases': [],       # Bias values at each iteration
            'losses': [],       # Loss values at each iteration
            'iterations': []    # Iteration numbers
        }
    
    def predict(self, X):
        """
        Make predictions using the linear model.
        
        Formula: y_pred = w * X + b
        
        This is the forward pass of the model.
        
        Args:
            X: Input features, shape (n_samples,) or (n_samples, 1)
            
        Returns:
            Predicted values, same shape as X
        """
        # Linear equation: y = mx + c (or y = wx + b in ML notation)
        # np.dot handles both scalar and array multiplication
        # For 1D linear regression: prediction = weight * input + bias
        return self.w * X + self.b
    
    def compute_loss(self, X, y):
        """
        Compute Mean Squared Error (MSE) loss.
        
        Formula: MSE = (1/n) * Σ(y_pred - y_actual)²
        
        The loss measures how far predictions are from actual values.
        Lower loss = better model fit.
        
        Args:
            X: Input features, shape (n_samples,)
            y: Actual target values, shape (n_samples,)
            
        Returns:
            MSE loss (single scalar value)
        """
        # Get number of samples
        # len(X) returns the number of data points
        n_samples = len(X)
        
        # Make predictions using current weights
        # y_pred shape: (n_samples,)
        y_pred = self.predict(X)
        
        # Compute squared differences
        # (y_pred - y) computes element-wise difference
        # ** 2 squares each difference (makes all values positive)
        squared_errors = (y_pred - y) ** 2
        
        # Compute mean of squared errors
        # np.mean averages all squared errors
        # This gives us a single number representing model performance
        mse = np.mean(squared_errors)
        
        return mse
    
    def compute_gradients(self, X, y):
        """
        Compute gradients of the loss function with respect to w and b.
        
        Using calculus, we derive:
        - ∂MSE/∂w = (2/n) * Σ(y_pred - y) * x
        - ∂MSE/∂b = (2/n) * Σ(y_pred - y)
        
        The gradient tells us:
        - Direction: which way to move w and b to reduce loss
        - Magnitude: how much the loss changes with small changes in w and b
        
        Args:
            X: Input features, shape (n_samples,)
            y: Actual target values, shape (n_samples,)
            
        Returns:
            dw: Gradient with respect to weight
            db: Gradient with respect to bias
        """
        # Get number of samples
        n_samples = len(X)
        
        # Make predictions with current parameters
        y_pred = self.predict(X)
        
        # Compute prediction errors (residuals)
        # Positive error: prediction too high
        # Negative error: prediction too low
        errors = y_pred - y  # Shape: (n_samples,)
        
        # ==================== GRADIENT CALCULATIONS ====================
        
        # Gradient of MSE with respect to weight (w)
        # Mathematical derivation:
        # MSE = (1/n) * Σ(wx + b - y)²
        # ∂MSE/∂w = (2/n) * Σ(wx + b - y) * x
        #         = (2/n) * Σ(error * x)
        # np.dot computes: Σ(errors[i] * X[i])
        dw = (2 / n_samples) * np.dot(errors, X)
        
        # Gradient of MSE with respect to bias (b)
        # Mathematical derivation:
        # ∂MSE/∂b = (2/n) * Σ(wx + b - y) * 1
        #         = (2/n) * Σ(error)
        # np.sum adds up all errors
        db = (2 / n_samples) * np.sum(errors)
        
        return dw, db
    
    def train_step(self, X, y):
        """
        Perform one iteration of gradient descent.
        
        This is the core of the training loop:
        1. Compute current loss
        2. Compute gradients
        3. Update weights and bias in direction that reduces loss
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            loss: Current loss value
        """
        # Step 1: Compute current loss
        # This tells us how well the model is doing
        loss = self.compute_loss(X, y)
        
        # Step 2: Compute gradients
        # Gradients tell us the direction and magnitude of change
        dw, db = self.compute_gradients(X, y)
        
        # Step 3: Update parameters using gradient descent
        # w_new = w_old - learning_rate * gradient
        # The negative sign moves us in the direction of decreasing loss
        # If gradient is positive, we decrease w
        # If gradient is negative, we increase w
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
        
        return loss
    
    def fit(self, X, y, iterations=1000, verbose=True, save_every=1):
        """
        Train the model using gradient descent for a specified number of iterations.
        
        Args:
            X: Training input features
            y: Training target values
            iterations: Number of gradient descent iterations
            verbose: Whether to print progress
            save_every: Save history every N iterations
            
        Returns:
            self (allows method chaining)
        """
        if verbose:
            print("=" * 60)
            print("   Training Linear Regression with Gradient Descent")
            print("=" * 60)
            print(f"   Learning rate: {self.learning_rate}")
            print(f"   Iterations: {iterations}")
            print(f"   Initial weight (w): {self.w:.4f}")
            print(f"   Initial bias (b): {self.b:.4f}")
            print("-" * 60)
        
        # Training loop
        for i in range(iterations):
            # Perform one training step
            loss = self.train_step(X, y)
            
            # Save history for visualization
            if i % save_every == 0:
                self.history['iterations'].append(i)
                self.history['weights'].append(self.w)
                self.history['biases'].append(self.b)
                self.history['losses'].append(loss)
            
            # Print progress at key iterations
            if verbose and (i + 1) in [1, 2, 10, 50, 100, 200, 400, 600, 800, iterations]:
                print(f"   Iteration {i + 1:4d} | Loss: {loss:.6f} | w: {self.w:.4f} | b: {self.b:.4f}")
        
        if verbose:
            print("-" * 60)
            print(f"   Final weight (w): {self.w:.4f}")
            print(f"   Final bias (b): {self.b:.4f}")
            print(f"   Final loss: {self.history['losses'][-1]:.6f}")
            print("=" * 60)
        
        return self
    
    def check_convergence(self, threshold=0.0001, window=50):
        """
        Check if the model has converged.
        
        Convergence is detected when the loss change is minimal over recent iterations.
        
        Args:
            threshold: Maximum allowed change in loss to consider converged
            window: Number of recent iterations to check
            
        Returns:
            (is_converged, convergence_iteration)
        """
        losses = self.history['losses']
        iterations = self.history['iterations']
        
        if len(losses) < window:
            return False, None
        
        # Check loss change over window
        for i in range(window, len(losses)):
            # Compute loss change in window
            loss_change = abs(losses[i] - losses[i - window])
            
            if loss_change < threshold:
                return True, iterations[i]
        
        return False, None


def generate_sample_data(n_samples=100, noise_level=2.0, seed=42):
    """
    Generate synthetic data for linear regression demonstration.
    
    Creates data that follows: y = true_w * x + true_b + noise
    
    This simulates real-world data where there's a linear relationship
    with some random variation (noise).
    
    Args:
        n_samples: Number of data points to generate
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
        
    Returns:
        X: Input features, shape (n_samples,)
        y: Target values, shape (n_samples,)
        true_w: True weight used to generate data
        true_b: True bias used to generate data
    """
    # Set random seed for reproducibility
    # This ensures we get the same "random" numbers each time
    np.random.seed(seed)
    
    # Define true parameters (what we're trying to learn)
    # Example: Miles per gallon vs car weight
    true_w = -5.44   # Weight (negative correlation)
    true_b = 35.94   # Bias (y-intercept)
    
    # Generate input features (e.g., car weight in 1000s of pounds)
    # np.random.uniform generates values uniformly between 2 and 5
    X = np.random.uniform(2, 5, n_samples)
    
    # Generate target values with noise
    # True relationship + random Gaussian noise
    # np.random.randn generates samples from standard normal
    # Multiply by noise_level to control noise magnitude
    noise = np.random.randn(n_samples) * noise_level
    y = true_w * X + true_b + noise
    
    return X, y, true_w, true_b


def compute_loss_surface(X, y, w_range, b_range):
    """
    Compute the loss for a grid of weight and bias values.
    
    This creates the data for the 3D loss surface visualization.
    
    Args:
        X: Input features
        y: Target values
        w_range: Array of weight values to evaluate
        b_range: Array of bias values to evaluate
        
    Returns:
        W: 2D grid of weight values
        B: 2D grid of bias values
        L: 2D grid of loss values
    """
    # Create a meshgrid of w and b values
    # np.meshgrid creates 2D coordinate matrices from 1D arrays
    W, B = np.meshgrid(w_range, b_range)
    
    # Initialize loss grid with same shape
    L = np.zeros_like(W)
    
    # Compute loss for each (w, b) combination
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w = W[i, j]
            b = B[i, j]
            # Prediction: y_pred = w * X + b
            y_pred = w * X + b
            # MSE loss
            L[i, j] = np.mean((y_pred - y) ** 2)
    
    return W, B, L


def plot_loss_curve(model, save_path='gradient_descent_loss_curve.png'):
    """
    Plot the loss curve showing how loss decreases during training.
    
    From Google ML Course:
    "Loss dramatically decreases during the first few iterations, 
    then gradually decreases before flattening out."
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Create figure
    plt.figure(figsize=(12, 5))
    
    # Get history data
    iterations = model.history['iterations']
    losses = model.history['losses']
    
    # Plot loss curve
    plt.plot(iterations, losses, color='#e74c3c', linewidth=2, label='Training Loss')
    
    # Mark convergence points
    # Find where loss reduction becomes minimal
    convergence_threshold = 0.01
    for i in range(1, len(losses)):
        if i > 50 and abs(losses[i] - losses[i-50]) < convergence_threshold:
            plt.axvline(x=iterations[i], color='#27ae60', linestyle='--', 
                       label=f'Approximate Convergence (~{iterations[i]} iterations)')
            break
    
    # Formatting
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Loss Curve: Model Convergence During Training', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotation
    plt.annotate('Steep decline\n(rapid learning)', 
                xy=(20, losses[min(20, len(losses)-1)]), 
                xytext=(100, losses[0]*0.8),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_3d_loss_surface(X, y, model, save_path='loss_surface_3d.png'):
    """
    Plot the 3D convex loss surface.
    
    From Google ML Course:
    "The loss functions for linear models always produce a convex surface.
    As a result, when a linear regression model converges, we know the model 
    has found the weights and bias that produce the lowest loss."
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Define range around the optimal point
    w_range = np.linspace(-10, 0, 50)
    b_range = np.linspace(20, 50, 50)
    
    # Compute loss surface
    W, B, L = compute_loss_surface(X, y, w_range, b_range)
    
    # Create 3D figure
    fig = plt.figure(figsize=(14, 6))
    
    # Subplot 1: 3D Surface
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot surface
    surf = ax1.plot_surface(W, B, L, cmap='viridis', alpha=0.8, 
                            linewidth=0, antialiased=True)
    
    # Plot gradient descent path
    weights = model.history['weights']
    biases = model.history['biases']
    losses = model.history['losses']
    
    # Sample every 10th point for cleaner visualization
    step = max(1, len(weights) // 50)
    ax1.plot(weights[::step], biases[::step], losses[::step], 
            'r.-', linewidth=2, markersize=4, label='Gradient Descent Path')
    
    # Mark start and end points
    ax1.scatter([weights[0]], [biases[0]], [losses[0]], 
               color='blue', s=100, marker='o', label='Start')
    ax1.scatter([weights[-1]], [biases[-1]], [losses[-1]], 
               color='green', s=100, marker='*', label='End (Minimum)')
    
    ax1.set_xlabel('Weight (w)', fontsize=10)
    ax1.set_ylabel('Bias (b)', fontsize=10)
    ax1.set_zlabel('Loss (MSE)', fontsize=10)
    ax1.set_title('3D Loss Surface (Convex)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # Subplot 2: Contour plot (top-down view)
    ax2 = fig.add_subplot(122)
    
    # Plot contours
    contour = ax2.contour(W, B, L, levels=30, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    
    # Plot gradient descent path
    ax2.plot(weights[::step], biases[::step], 'r.-', 
            linewidth=2, markersize=4, label='Gradient Descent Path')
    ax2.scatter([weights[0]], [biases[0]], color='blue', s=100, 
               marker='o', zorder=5, label='Start')
    ax2.scatter([weights[-1]], [biases[-1]], color='green', s=100, 
               marker='*', zorder=5, label='End (Minimum)')
    
    ax2.set_xlabel('Weight (w)', fontsize=10)
    ax2.set_ylabel('Bias (b)', fontsize=10)
    ax2.set_title('Contour Plot (Top-Down View)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_snapshots(X, y, model, true_w, true_b, save_path='model_snapshots.png'):
    """
    Plot the model at different stages of training.
    
    From Google ML Course:
    "Visualizing the model's state at snapshots during the training process 
    solidifies the link between updating the weights and bias, reducing loss, 
    and model convergence."
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Get history
    iterations = model.history['iterations']
    weights = model.history['weights']
    biases = model.history['biases']
    losses = model.history['losses']
    
    # Select snapshots: beginning, middle, end
    n = len(iterations)
    snapshot_indices = [1, n // 3, n - 1]  # Early, mid, late
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # X range for plotting the line
    X_line = np.linspace(X.min() - 0.5, X.max() + 0.5, 100)
    
    titles = ['Beginning of Training', 'Midway Through Training', 'End of Training (Converged)']
    
    for ax, idx, title in zip(axes, snapshot_indices, titles):
        # Get parameters at this snapshot
        w = weights[idx]
        b = biases[idx]
        loss = losses[idx]
        iteration = iterations[idx]
        
        # Plot data points
        ax.scatter(X, y, color='#3498db', alpha=0.6, s=50, label='Data Points')
        
        # Plot model line
        y_line = w * X_line + b
        ax.plot(X_line, y_line, color='#e74c3c', linewidth=2, 
               label=f'Model (w={w:.2f}, b={b:.2f})')
        
        # Plot true line for reference
        y_true_line = true_w * X_line + true_b
        ax.plot(X_line, y_true_line, color='#27ae60', linewidth=2, 
               linestyle='--', alpha=0.7, label=f'True (w={true_w}, b={true_b})')
        
        # Draw loss lines (residuals) for a few points
        sample_indices = np.random.choice(len(X), min(10, len(X)), replace=False)
        for i in sample_indices:
            y_pred = w * X[i] + b
            ax.plot([X[i], X[i]], [y[i], y_pred], color='gray', 
                   linestyle='-', alpha=0.5, linewidth=1)
        
        # Formatting
        ax.set_xlabel('X (Input Feature)', fontsize=10)
        ax.set_ylabel('Y (Target)', fontsize=10)
        ax.set_title(f'{title}\nIteration: {iteration}, Loss: {loss:.4f}', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Model Evolution During Gradient Descent Training', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_combined_visualization(X, y, model, true_w, true_b, 
                               save_path='gradient_descent_visualization.png'):
    """
    Create a comprehensive visualization combining loss curve and model fit.
    
    This mimics the diagrams from the Google ML Crash Course.
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Get history
    iterations = model.history['iterations']
    losses = model.history['losses']
    weights = model.history['weights']
    biases = model.history['biases']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    
    # ==================== SUBPLOT 1: LOSS CURVE ====================
    ax1 = fig.add_subplot(221)
    ax1.plot(iterations, losses, color='#e74c3c', linewidth=2)
    ax1.set_xlabel('Iterations', fontsize=11)
    ax1.set_ylabel('Loss (MSE)', fontsize=11)
    ax1.set_title('Loss Curve During Training', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Mark key points
    key_iterations = [0, len(iterations)//4, len(iterations)//2, len(iterations)-1]
    for ki in key_iterations:
        if ki < len(iterations):
            ax1.scatter(iterations[ki], losses[ki], s=100, zorder=5)
            ax1.annotate(f'Iter {iterations[ki]}', 
                        (iterations[ki], losses[ki]),
                        textcoords="offset points", xytext=(10, 10),
                        fontsize=9)
    
    # ==================== SUBPLOT 2: PARAMETER EVOLUTION ====================
    ax2 = fig.add_subplot(222)
    ax2.plot(iterations, weights, color='#3498db', linewidth=2, label=f'Weight (w) → {weights[-1]:.2f}')
    ax2.plot(iterations, biases, color='#27ae60', linewidth=2, label=f'Bias (b) → {biases[-1]:.2f}')
    ax2.axhline(y=true_w, color='#3498db', linestyle='--', alpha=0.5, label=f'True w = {true_w}')
    ax2.axhline(y=true_b, color='#27ae60', linestyle='--', alpha=0.5, label=f'True b = {true_b}')
    ax2.set_xlabel('Iterations', fontsize=11)
    ax2.set_ylabel('Parameter Value', fontsize=11)
    ax2.set_title('Weight and Bias Evolution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ==================== SUBPLOT 3: FINAL MODEL FIT ====================
    ax3 = fig.add_subplot(223)
    
    # Plot data
    ax3.scatter(X, y, color='#3498db', alpha=0.6, s=50, label='Training Data')
    
    # Plot model line
    X_line = np.linspace(X.min() - 0.5, X.max() + 0.5, 100)
    y_pred = weights[-1] * X_line + biases[-1]
    ax3.plot(X_line, y_pred, color='#e74c3c', linewidth=2, 
            label=f'Learned Model: y = {weights[-1]:.2f}x + {biases[-1]:.2f}')
    
    # Plot true line
    y_true = true_w * X_line + true_b
    ax3.plot(X_line, y_true, color='#27ae60', linewidth=2, linestyle='--',
            label=f'True Model: y = {true_w}x + {true_b}')
    
    ax3.set_xlabel('X (Input Feature)', fontsize=11)
    ax3.set_ylabel('Y (Target)', fontsize=11)
    ax3.set_title('Final Model Fit', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ==================== SUBPLOT 4: GRADIENT DESCENT EXPLANATION ====================
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    explanation = """
    GRADIENT DESCENT ALGORITHM
    ==========================
    
    1. INITIALIZE: Start with random weights and bias near zero
       w = {:.4f}, b = {:.4f}
    
    2. ITERATE: Repeat the following steps:
       
       a) COMPUTE LOSS: Calculate how wrong predictions are
          Loss = (1/n) * Σ(y_pred - y_actual)²
       
       b) COMPUTE GRADIENTS: Find direction to improve
          ∂Loss/∂w = (2/n) * Σ(error * x)
          ∂Loss/∂b = (2/n) * Σ(error)
       
       c) UPDATE PARAMETERS: Move in direction of lower loss
          w = w - learning_rate * ∂Loss/∂w
          b = b - learning_rate * ∂Loss/∂b
    
    3. CONVERGENCE: Stop when loss stops decreasing
       Final w = {:.4f}, Final b = {:.4f}
       Final Loss = {:.6f}
    
    KEY INSIGHT: Linear models have CONVEX loss surfaces,
    guaranteeing gradient descent finds the GLOBAL minimum!
    """.format(
        model.history['weights'][0], model.history['biases'][0],
        weights[-1], biases[-1], losses[-1]
    )
    
    ax4.text(0.05, 0.95, explanation, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.suptitle('Linear Regression with Gradient Descent\n(Based on Google ML Crash Course)',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_gradient_tangent_lines(X, y, model, save_path='gradient_tangent_lines.png'):
    """
    Visualize gradients as tangent lines on the loss curves.
    
    This demonstrates the geometric meaning of gradients:
    - The gradient is the SLOPE of the tangent line at a point
    - Positive gradient = loss increases as parameter increases
    - Negative gradient = loss decreases as parameter increases
    - Gradient descent moves OPPOSITE to the gradient direction
    
    We show two plots:
    1. Loss vs Weight (with bias fixed) - showing dL/dw as tangent slope
    2. Loss vs Bias (with weight fixed) - showing dL/db as tangent slope
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Get current parameters at a specific iteration
    # Use an early iteration where gradients are large and visible
    iteration_idx = min(10, len(model.history['weights']) - 1)
    current_w = model.history['weights'][iteration_idx]
    current_b = model.history['biases'][iteration_idx]
    
    # Compute the current gradient at this point
    n_samples = len(X)
    y_pred = current_w * X + current_b
    errors = y_pred - y
    
    # Gradient with respect to weight: dL/dw = (2/n) * Σ(error * x)
    dw = (2 / n_samples) * np.dot(errors, X)
    
    # Gradient with respect to bias: dL/db = (2/n) * Σ(error)
    db = (2 / n_samples) * np.sum(errors)
    
    # Current loss
    current_loss = np.mean(errors ** 2)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ==================== SUBPLOT 1: Loss vs Weight ====================
    ax1 = axes[0]
    
    # Create range of weight values around current weight
    w_range = np.linspace(current_w - 5, current_w + 5, 100)
    
    # Compute loss for each weight (holding bias constant)
    losses_w = []
    for w in w_range:
        y_pred_temp = w * X + current_b
        loss_temp = np.mean((y_pred_temp - y) ** 2)
        losses_w.append(loss_temp)
    losses_w = np.array(losses_w)
    
    # Plot the loss curve
    ax1.plot(w_range, losses_w, 'b-', linewidth=2.5, label='Loss L(w) with b fixed')
    
    # Mark the current point
    ax1.scatter([current_w], [current_loss], color='red', s=150, zorder=5, 
               label=f'Current point (w={current_w:.2f})')
    
    # Draw tangent line at current point
    # Tangent line equation: y - y0 = m(x - x0), where m is the gradient
    # y = gradient * (x - current_w) + current_loss
    tangent_w_range = np.linspace(current_w - 2, current_w + 2, 50)
    tangent_line_w = dw * (tangent_w_range - current_w) + current_loss
    
    ax1.plot(tangent_w_range, tangent_line_w, 'r--', linewidth=2.5, 
            label=f'Tangent line (slope = dL/dw = {dw:.2f})')
    
    # Add arrow showing gradient direction and update direction
    arrow_start_w = current_w
    arrow_end_w = current_w - 0.5 * np.sign(dw)  # Opposite to gradient
    ax1.annotate('', xy=(arrow_end_w, current_loss - 5), 
                xytext=(arrow_start_w, current_loss),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax1.text(current_w - 0.5, current_loss - 8, 'Update\ndirection', 
            fontsize=10, ha='center', color='green', fontweight='bold')
    
    # Add annotation for gradient
    ax1.annotate(f'Gradient dL/dw = {dw:.2f}\n(slope of tangent)',
                xy=(current_w + 0.5, current_loss + dw * 0.5),
                xytext=(current_w + 2, current_loss + 20),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Weight (w)', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Loss vs Weight: Gradient as Tangent Slope\n'
                 f'∂L/∂w = (2/n) × Σ(error × x) = {dw:.4f}', 
                 fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # ==================== SUBPLOT 2: Loss vs Bias ====================
    ax2 = axes[1]
    
    # Create range of bias values around current bias
    b_range = np.linspace(current_b - 15, current_b + 15, 100)
    
    # Compute loss for each bias (holding weight constant)
    losses_b = []
    for b in b_range:
        y_pred_temp = current_w * X + b
        loss_temp = np.mean((y_pred_temp - y) ** 2)
        losses_b.append(loss_temp)
    losses_b = np.array(losses_b)
    
    # Plot the loss curve
    ax2.plot(b_range, losses_b, 'b-', linewidth=2.5, label='Loss L(b) with w fixed')
    
    # Mark the current point
    ax2.scatter([current_b], [current_loss], color='red', s=150, zorder=5,
               label=f'Current point (b={current_b:.2f})')
    
    # Draw tangent line at current point
    tangent_b_range = np.linspace(current_b - 5, current_b + 5, 50)
    tangent_line_b = db * (tangent_b_range - current_b) + current_loss
    
    ax2.plot(tangent_b_range, tangent_line_b, 'r--', linewidth=2.5,
            label=f'Tangent line (slope = dL/db = {db:.2f})')
    
    # Add arrow showing gradient direction and update direction
    arrow_start_b = current_b
    arrow_end_b = current_b - 2 * np.sign(db)  # Opposite to gradient
    ax2.annotate('', xy=(arrow_end_b, current_loss - 10), 
                xytext=(arrow_start_b, current_loss),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax2.text(current_b - 2, current_loss - 15, 'Update\ndirection', 
            fontsize=10, ha='center', color='green', fontweight='bold')
    
    # Add annotation for gradient
    ax2.annotate(f'Gradient dL/db = {db:.2f}\n(slope of tangent)',
                xy=(current_b + 1, current_loss + db * 1),
                xytext=(current_b + 8, current_loss + 30),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_xlabel('Bias (b)', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.set_title('Loss vs Bias: Gradient as Tangent Slope\n'
                 f'∂L/∂b = (2/n) × Σ(error) = {db:.4f}', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.suptitle('Visualizing Gradients: The Tangent Line Shows the Direction of Steepest Ascent\n'
                'Gradient Descent moves OPPOSITE to the gradient (downhill)',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_gradient_evolution(X, y, model, save_path='gradient_evolution.png'):
    """
    Show how gradients change during training with tangent lines at multiple points.
    
    This demonstrates:
    - Gradients are large at the start (steep tangent)
    - Gradients become smaller near the minimum (flat tangent)
    - At the minimum, gradients are approximately zero
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Select iterations to visualize (early, mid, late)
    n_iters = len(model.history['weights'])
    indices = [0, n_iters // 4, n_iters // 2, 3 * n_iters // 4, n_iters - 1]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Get optimal weight and bias for reference
    optimal_w = model.history['weights'][-1]
    optimal_b = model.history['biases'][-1]
    
    # ==================== TOP ROW: Loss vs Weight at different iterations ====================
    
    # Create weight range for plotting
    w_range = np.linspace(-8, 6, 200)
    
    for ax_idx, iter_idx in enumerate(indices[:3]):
        ax = axes[0, ax_idx]
        
        current_w = model.history['weights'][iter_idx]
        current_b = model.history['biases'][iter_idx]
        iteration = model.history['iterations'][iter_idx]
        
        # Compute loss curve for this bias
        losses = []
        for w in w_range:
            y_pred = w * X + current_b
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)
        losses = np.array(losses)
        
        # Current loss
        y_pred_current = current_w * X + current_b
        errors = y_pred_current - y
        current_loss = np.mean(errors ** 2)
        
        # Compute gradient at current point
        dw = (2 / len(X)) * np.dot(errors, X)
        
        # Plot loss curve
        ax.plot(w_range, losses, 'b-', linewidth=2, label='Loss curve L(w)')
        
        # Mark current point
        ax.scatter([current_w], [current_loss], color='red', s=120, zorder=5)
        
        # Draw tangent line
        tangent_range = np.linspace(current_w - 2.5, current_w + 2.5, 50)
        tangent = dw * (tangent_range - current_w) + current_loss
        ax.plot(tangent_range, tangent, 'r--', linewidth=2.5, 
               label=f'Tangent (slope={dw:.1f})')
        
        # Mark optimal point
        y_pred_opt = optimal_w * X + current_b
        opt_loss = np.mean((y_pred_opt - y) ** 2)
        ax.scatter([optimal_w], [opt_loss], color='green', s=100, marker='*', 
                  zorder=5, label='Optimal w')
        
        ax.set_xlabel('Weight (w)', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(f'Iteration {iteration}\nw={current_w:.2f}, dL/dw={dw:.2f}', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(losses) * 1.1)
    
    # ==================== BOTTOM ROW: Loss vs Bias at different iterations ====================
    
    # Create bias range for plotting
    b_range = np.linspace(-5, 50, 200)
    
    for ax_idx, iter_idx in enumerate(indices[:3]):
        ax = axes[1, ax_idx]
        
        current_w = model.history['weights'][iter_idx]
        current_b = model.history['biases'][iter_idx]
        iteration = model.history['iterations'][iter_idx]
        
        # Compute loss curve for this weight
        losses = []
        for b in b_range:
            y_pred = current_w * X + b
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)
        losses = np.array(losses)
        
        # Current loss
        y_pred_current = current_w * X + current_b
        errors = y_pred_current - y
        current_loss = np.mean(errors ** 2)
        
        # Compute gradient at current point
        db = (2 / len(X)) * np.sum(errors)
        
        # Plot loss curve
        ax.plot(b_range, losses, 'b-', linewidth=2, label='Loss curve L(b)')
        
        # Mark current point
        ax.scatter([current_b], [current_loss], color='red', s=120, zorder=5)
        
        # Draw tangent line
        tangent_range = np.linspace(current_b - 8, current_b + 8, 50)
        tangent = db * (tangent_range - current_b) + current_loss
        # Clip tangent to reasonable values
        tangent = np.clip(tangent, 0, max(losses) * 1.5)
        ax.plot(tangent_range, tangent, 'r--', linewidth=2.5, 
               label=f'Tangent (slope={db:.1f})')
        
        # Mark optimal point
        y_pred_opt = current_w * X + optimal_b
        opt_loss = np.mean((y_pred_opt - y) ** 2)
        ax.scatter([optimal_b], [opt_loss], color='green', s=100, marker='*', 
                  zorder=5, label='Optimal b')
        
        ax.set_xlabel('Bias (b)', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(f'Iteration {iteration}\nb={current_b:.2f}, dL/db={db:.2f}', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(losses) * 1.1)
    
    plt.suptitle('Gradient Evolution: Tangent Lines Become Flatter Near the Minimum\n'
                '(Steep tangent = large gradient = far from minimum | '
                'Flat tangent = small gradient = near minimum)',
                fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_first_10_iterations(X, y, model, true_w, true_b, save_path='first_10_iterations.png'):
    """
    Show the predicted line vs actual (true) line for the first 10 iterations.
    
    This demonstrates how gradient descent progressively improves the model:
    - Iteration 0: Random starting point (poor fit)
    - Each iteration: Model gets closer to the true line
    - Watch the predicted line rotate and shift toward the optimal position
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Create figure with 10 subplots (2 rows x 5 columns)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()  # Flatten to 1D array for easy indexing
    
    # X range for plotting lines
    X_line = np.linspace(X.min() - 0.3, X.max() + 0.3, 100)
    
    # True line (what we're trying to learn)
    y_true_line = true_w * X_line + true_b
    
    # Get history data
    weights = model.history['weights']
    biases = model.history['biases']
    losses = model.history['losses']
    iterations = model.history['iterations']
    
    # Plot first 10 iterations
    for i in range(10):
        ax = axes[i]
        
        # Get parameters at this iteration
        if i < len(weights):
            w = weights[i]
            b = biases[i]
            loss = losses[i]
            iteration = iterations[i]
        else:
            # If we don't have 10 iterations saved, use the last one
            w = weights[-1]
            b = biases[-1]
            loss = losses[-1]
            iteration = iterations[-1]
        
        # Predicted line at this iteration
        y_pred_line = w * X_line + b
        
        # Plot data points
        ax.scatter(X, y, color='#3498db', alpha=0.5, s=30, label='Data')
        
        # Plot true line (target)
        ax.plot(X_line, y_true_line, 'g--', linewidth=2, 
               label=f'True: y={true_w}x+{true_b}')
        
        # Plot predicted line
        ax.plot(X_line, y_pred_line, 'r-', linewidth=2.5, 
               label=f'Pred: y={w:.2f}x+{b:.2f}')
        
        # Calculate and show error lines for a few points
        sample_idx = np.random.choice(len(X), min(5, len(X)), replace=False)
        for idx in sample_idx:
            y_pred_point = w * X[idx] + b
            ax.plot([X[idx], X[idx]], [y[idx], y_pred_point], 
                   'gray', alpha=0.4, linewidth=1)
        
        # Formatting
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_title(f'Iteration {iteration}\n'
                    f'w={w:.2f}, b={b:.2f}\n'
                    f'Loss={loss:.2f}', 
                    fontsize=10, fontweight='bold')
        
        # Set consistent y-limits for comparison
        y_min = min(y.min(), y_true_line.min()) - 5
        y_max = max(y.max(), y_true_line.max()) + 5
        ax.set_ylim(y_min, y_max)
        
        ax.grid(True, alpha=0.3)
        
        # Only show legend for first subplot
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    plt.suptitle('Predicted Line vs True Line: First 10 Iterations of Gradient Descent\n'
                '(Red = Predicted, Green dashed = True, Gray lines = Errors)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_iteration_animation_frames(X, y, model, true_w, true_b, 
                                    save_path='iteration_frames.png'):
    """
    Create a larger visualization showing iterations 1, 2, 5, 10 with detailed info.
    
    Shows:
    - The model line at each iteration
    - How parameters (w, b) change
    - The residual errors
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Select key iterations to visualize
    key_iterations = [0, 1, 2, 5, 10, 50]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # X range for plotting
    X_line = np.linspace(X.min() - 0.5, X.max() + 0.5, 100)
    y_true_line = true_w * X_line + true_b
    
    # Get history
    weights = model.history['weights']
    biases = model.history['biases']
    losses = model.history['losses']
    iterations = model.history['iterations']
    
    # Color gradient from red (start) to green (end)
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60', '#1abc9c']
    
    for ax_idx, target_iter in enumerate(key_iterations):
        ax = axes[ax_idx]
        
        # Find the closest iteration in our history
        iter_idx = min(target_iter, len(weights) - 1)
        
        w = weights[iter_idx]
        b = biases[iter_idx]
        loss = losses[iter_idx]
        
        # Predicted line
        y_pred_line = w * X_line + b
        
        # Plot data points
        ax.scatter(X, y, color='#3498db', alpha=0.6, s=60, 
                  label='Training Data', zorder=3)
        
        # Plot true line
        ax.plot(X_line, y_true_line, 'g--', linewidth=2.5, 
               label=f'True Line: y = {true_w}x + {true_b}', zorder=2)
        
        # Plot predicted line with iteration-specific color
        ax.plot(X_line, y_pred_line, color=colors[ax_idx], linewidth=3, 
               label=f'Predicted: y = {w:.2f}x + {b:.2f}', zorder=4)
        
        # Draw all error lines (residuals)
        y_pred_points = w * X + b
        for j in range(len(X)):
            ax.plot([X[j], X[j]], [y[j], y_pred_points[j]], 
                   color='gray', alpha=0.3, linewidth=1, zorder=1)
        
        # Calculate metrics
        mae = np.mean(np.abs(y_pred_points - y))
        
        # Add text box with details
        textstr = f'Weight (w): {w:.4f}\n'
        textstr += f'Bias (b): {b:.4f}\n'
        textstr += f'MSE Loss: {loss:.4f}\n'
        textstr += f'MAE: {mae:.4f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        # Formatting
        ax.set_xlabel('X (Input Feature)', fontsize=11)
        ax.set_ylabel('Y (Target)', fontsize=11)
        ax.set_title(f'Iteration {target_iter}', fontsize=13, fontweight='bold',
                    color=colors[ax_idx])
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Consistent limits
        y_min = min(y.min(), y_true_line.min()) - 3
        y_max = max(y.max(), y_true_line.max()) + 3
        ax.set_ylim(y_min, y_max)
    
    plt.suptitle('Model Evolution: Predicted Line vs True Line Across Training\n'
                'Watch the red line (predicted) converge toward the green dashed line (true)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_parameter_trajectory(X, y, model, true_w, true_b, 
                              save_path='parameter_trajectory.png'):
    """
    Show how (w, b) move through parameter space during first 10 iterations.
    
    Combines:
    - Parameter space trajectory (how w, b change)
    - Model visualization at each step
    """
    if not HAS_MATPLOTLIB:
        return
    
    fig = plt.figure(figsize=(18, 6))
    
    # Get history
    weights = model.history['weights'][:11]  # First 11 iterations
    biases = model.history['biases'][:11]
    losses = model.history['losses'][:11]
    
    # ==================== SUBPLOT 1: Parameter Space Trajectory ====================
    ax1 = fig.add_subplot(131)
    
    # Create contour plot of loss surface
    w_range = np.linspace(min(weights) - 2, max(weights) + 2, 50)
    b_range = np.linspace(min(biases) - 5, max(biases) + 5, 50)
    W, B = np.meshgrid(w_range, b_range)
    
    L = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            y_pred = W[i, j] * X + B[i, j]
            L[i, j] = np.mean((y_pred - y) ** 2)
    
    contour = ax1.contour(W, B, L, levels=20, cmap='viridis', alpha=0.7)
    ax1.clabel(contour, inline=True, fontsize=8)
    
    # Plot trajectory with numbered points
    for i in range(len(weights)):
        color = plt.cm.Reds(i / len(weights))
        ax1.scatter(weights[i], biases[i], color=color, s=150, zorder=5, 
                   edgecolors='black', linewidth=1)
        ax1.annotate(str(i), (weights[i], biases[i]), fontsize=10, 
                    ha='center', va='center', fontweight='bold', color='white')
    
    # Connect points with arrows
    for i in range(len(weights) - 1):
        ax1.annotate('', xy=(weights[i+1], biases[i+1]), 
                    xytext=(weights[i], biases[i]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Mark true parameters
    ax1.scatter([true_w], [true_b], color='green', s=200, marker='*', 
               zorder=6, label=f'True (w={true_w}, b={true_b})')
    
    ax1.set_xlabel('Weight (w)', fontsize=12)
    ax1.set_ylabel('Bias (b)', fontsize=12)
    ax1.set_title('Parameter Space: First 10 Steps\n(Numbers show iteration)', 
                 fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ==================== SUBPLOT 2: Weight Over Iterations ====================
    ax2 = fig.add_subplot(132)
    
    iterations = list(range(len(weights)))
    ax2.plot(iterations, weights, 'bo-', linewidth=2, markersize=10, label='Weight (w)')
    ax2.axhline(y=true_w, color='g', linestyle='--', linewidth=2, 
               label=f'True w = {true_w}')
    
    # Add tangent lines at each point showing the rate of change
    for i in range(len(weights)):
        ax2.annotate(f'{weights[i]:.1f}', (i, weights[i]), 
                    textcoords="offset points", xytext=(0, 10), fontsize=8)
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Weight (w)', fontsize=12)
    ax2.set_title('Weight Evolution: First 10 Iterations', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(iterations)
    
    # ==================== SUBPLOT 3: Bias Over Iterations ====================
    ax3 = fig.add_subplot(133)
    
    ax3.plot(iterations, biases, 'ro-', linewidth=2, markersize=10, label='Bias (b)')
    ax3.axhline(y=true_b, color='g', linestyle='--', linewidth=2, 
               label=f'True b = {true_b}')
    
    for i in range(len(biases)):
        ax3.annotate(f'{biases[i]:.1f}', (i, biases[i]), 
                    textcoords="offset points", xytext=(0, 10), fontsize=8)
    
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Bias (b)', fontsize=12)
    ax3.set_title('Bias Evolution: First 10 Iterations', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(iterations)
    
    plt.suptitle('Parameter Evolution During First 10 Gradient Descent Iterations\n'
                'Watch (w, b) move from random start toward optimal values',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_gradient_vector_field(X, y, model, save_path='gradient_vector_field.png'):
    """
    Plot a 2D vector field showing the gradient direction at each point.
    
    This visualizes:
    - Arrows pointing in gradient direction (uphill)
    - Gradient descent moves opposite to arrows (downhill)
    - Arrows converge toward the minimum
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Create grid of (w, b) values
    w_range = np.linspace(-8, 2, 15)
    b_range = np.linspace(10, 50, 15)
    W, B = np.meshgrid(w_range, b_range)
    
    # Compute gradients and loss at each point
    dW = np.zeros_like(W)
    dB = np.zeros_like(B)
    L = np.zeros_like(W)
    
    n_samples = len(X)
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w = W[i, j]
            b = B[i, j]
            
            # Prediction and error
            y_pred = w * X + b
            errors = y_pred - y
            
            # Loss
            L[i, j] = np.mean(errors ** 2)
            
            # Gradients
            dW[i, j] = (2 / n_samples) * np.dot(errors, X)
            dB[i, j] = (2 / n_samples) * np.sum(errors)
    
    # Normalize gradients for better visualization
    magnitude = np.sqrt(dW**2 + dB**2)
    dW_norm = dW / (magnitude + 1e-10)
    dB_norm = dB / (magnitude + 1e-10)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ==================== SUBPLOT 1: Vector Field with Contours ====================
    ax1 = axes[0]
    
    # Plot contours
    contour = ax1.contour(W, B, L, levels=20, cmap='viridis', alpha=0.7)
    ax1.clabel(contour, inline=True, fontsize=8)
    
    # Plot gradient vectors (pointing uphill - direction of increase)
    # We negate to show the update direction (downhill)
    ax1.quiver(W, B, -dW_norm, -dB_norm, magnitude, cmap='Reds', 
              scale=25, width=0.004, alpha=0.8)
    
    # Plot gradient descent path
    weights = model.history['weights']
    biases = model.history['biases']
    step = max(1, len(weights) // 30)
    ax1.plot(weights[::step], biases[::step], 'b.-', linewidth=2, 
            markersize=8, label='Gradient Descent Path')
    ax1.scatter([weights[0]], [biases[0]], color='blue', s=150, 
               marker='o', zorder=5, label='Start')
    ax1.scatter([weights[-1]], [biases[-1]], color='green', s=200, 
               marker='*', zorder=5, label='End (Minimum)')
    
    ax1.set_xlabel('Weight (w)', fontsize=12)
    ax1.set_ylabel('Bias (b)', fontsize=12)
    ax1.set_title('Gradient Vector Field\n'
                 'Arrows show update direction (opposite to gradient)',
                 fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ==================== SUBPLOT 2: Gradient Magnitude ====================
    ax2 = axes[1]
    
    # Plot gradient magnitude as heatmap
    im = ax2.pcolormesh(W, B, magnitude, cmap='hot', shading='auto')
    plt.colorbar(im, ax=ax2, label='Gradient Magnitude |∇L|')
    
    # Plot contours of loss
    contour2 = ax2.contour(W, B, L, levels=15, colors='white', alpha=0.5, linewidths=0.5)
    
    # Plot gradient descent path
    ax2.plot(weights[::step], biases[::step], 'c.-', linewidth=2, 
            markersize=8, label='Gradient Descent Path')
    ax2.scatter([weights[0]], [biases[0]], color='cyan', s=150, 
               marker='o', zorder=5, edgecolors='white', linewidth=2, label='Start')
    ax2.scatter([weights[-1]], [biases[-1]], color='lime', s=200, 
               marker='*', zorder=5, edgecolors='white', linewidth=2, label='End (Minimum)')
    
    ax2.set_xlabel('Weight (w)', fontsize=12)
    ax2.set_ylabel('Bias (b)', fontsize=12)
    ax2.set_title('Gradient Magnitude Heatmap\n'
                 '(Darker = smaller gradient = closer to minimum)',
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('Gradient Visualization: The Gradient Points Toward Steepest Increase\n'
                'Gradient Descent follows OPPOSITE direction (steepest DECREASE)',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_learning_rates():
    """
    Demonstrate the effect of different learning rates.
    
    Shows:
    - Too small: slow convergence
    - Just right: smooth convergence
    - Too large: may overshoot or diverge
    """
    if not HAS_MATPLOTLIB:
        return
    
    print("\n" + "=" * 60)
    print("   Demonstrating Effect of Learning Rate")
    print("=" * 60)
    
    # Generate data
    X, y, _, _ = generate_sample_data(n_samples=100)
    
    # Different learning rates
    learning_rates = [0.001, 0.01, 0.1]
    labels = ['Too Small (0.001)', 'Just Right (0.01)', 'Large (0.1)']
    colors = ['#e74c3c', '#27ae60', '#3498db']
    
    # Create figure
    plt.figure(figsize=(12, 5))
    
    for lr, label, color in zip(learning_rates, labels, colors):
        # Create and train model
        model = LinearRegressionGD(learning_rate=lr)
        model.fit(X, y, iterations=500, verbose=False)
        
        # Plot loss curve
        plt.plot(model.history['iterations'], model.history['losses'], 
                label=label, color=color, linewidth=2)
        
        print(f"   {label}: Final Loss = {model.history['losses'][-1]:.6f}")
    
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Effect of Learning Rate on Convergence', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function demonstrating Linear Regression with Gradient Descent.
    
    This follows the structure of Google's ML Crash Course.
    """
    print()
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "   LINEAR REGRESSION WITH GRADIENT DESCENT FROM SCRATCH".center(68) + "#")
    print("#" + "   Based on Google ML Crash Course".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print()
    
    # ==================== STEP 1: GENERATE DATA ====================
    print("STEP 1: Generating Synthetic Data")
    print("-" * 40)
    
    # Generate sample data
    # Using car weight vs MPG example from Google course
    X, y, true_w, true_b = generate_sample_data(n_samples=100, noise_level=2.0)
    
    print(f"   Generated {len(X)} samples")
    print(f"   True weight (w): {true_w}")
    print(f"   True bias (b): {true_b}")
    print(f"   X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   y range: [{y.min():.2f}, {y.max():.2f}]")
    print()
    
    # ==================== STEP 2: CREATE AND TRAIN MODEL ====================
    print("STEP 2: Training the Model")
    print("-" * 40)
    
    # Create model with learning rate
    model = LinearRegressionGD(learning_rate=0.05)
    
    # Train for 1000 iterations
    model.fit(X, y, iterations=1000, verbose=True)
    print()
    
    # ==================== STEP 3: CHECK CONVERGENCE ====================
    print("STEP 3: Checking Convergence")
    print("-" * 40)
    
    converged, convergence_iter = model.check_convergence(threshold=0.001, window=50)
    
    if converged:
        print(f"   Model CONVERGED around iteration {convergence_iter}")
        print("   Additional iterations won't significantly reduce loss.")
    else:
        print("   Model may need more iterations to fully converge.")
    
    # Compare learned vs true parameters
    print()
    print("   Parameter Comparison:")
    print(f"   {'Parameter':<12} {'True':<12} {'Learned':<12} {'Error':<12}")
    print(f"   {'-'*48}")
    print(f"   {'Weight (w)':<12} {true_w:<12.4f} {model.w:<12.4f} {abs(true_w - model.w):<12.4f}")
    print(f"   {'Bias (b)':<12} {true_b:<12.4f} {model.b:<12.4f} {abs(true_b - model.b):<12.4f}")
    print()
    
    # ==================== STEP 4: VISUALIZATIONS ====================
    if HAS_MATPLOTLIB:
        print("STEP 4: Creating Visualizations")
        print("-" * 40)
        
        # Plot loss curve
        print("   Creating loss curve...")
        plot_loss_curve(model)
        print("   Saved 'gradient_descent_loss_curve.png'")
        
        # Plot 3D loss surface
        print("   Creating 3D loss surface...")
        plot_3d_loss_surface(X, y, model)
        print("   Saved 'loss_surface_3d.png'")
        
        # Plot model snapshots
        print("   Creating model snapshots...")
        plot_model_snapshots(X, y, model, true_w, true_b)
        print("   Saved 'model_snapshots.png'")
        
        # Plot combined visualization
        print("   Creating combined visualization...")
        plot_combined_visualization(X, y, model, true_w, true_b)
        print("   Saved 'gradient_descent_visualization.png'")
        
        # ==================== NEW: GRADIENT TANGENT VISUALIZATIONS ====================
        
        # Plot gradient tangent lines (NEW)
        print("   Creating gradient tangent line visualization...")
        plot_gradient_tangent_lines(X, y, model)
        print("   Saved 'gradient_tangent_lines.png'")
        
        # Plot gradient evolution over training (NEW)
        print("   Creating gradient evolution visualization...")
        plot_gradient_evolution(X, y, model)
        print("   Saved 'gradient_evolution.png'")
        
        # Plot gradient vector field (NEW)
        print("   Creating gradient vector field visualization...")
        plot_gradient_vector_field(X, y, model)
        print("   Saved 'gradient_vector_field.png'")
        
        # ==================== NEW: FIRST 10 ITERATIONS VISUALIZATION ====================
        
        # Plot first 10 iterations (NEW)
        print("   Creating first 10 iterations visualization...")
        plot_first_10_iterations(X, y, model, true_w, true_b)
        print("   Saved 'first_10_iterations.png'")
        
        # Plot iteration frames with details (NEW)
        print("   Creating iteration frames visualization...")
        plot_iteration_animation_frames(X, y, model, true_w, true_b)
        print("   Saved 'iteration_frames.png'")
        
        # Plot parameter trajectory (NEW)
        print("   Creating parameter trajectory visualization...")
        plot_parameter_trajectory(X, y, model, true_w, true_b)
        print("   Saved 'parameter_trajectory.png'")
        
        # Demonstrate learning rates
        demonstrate_learning_rates()
        print("   Saved 'learning_rate_comparison.png'")
        print()
    
    # ==================== STEP 5: SUMMARY ====================
    print("=" * 70)
    print("   SUMMARY: KEY CONCEPTS")
    print("=" * 70)
    print("""
    1. GRADIENT DESCENT is an iterative optimization algorithm that:
       - Starts with random weights and bias
       - Computes gradients (direction of steepest increase in loss)
       - Updates parameters in opposite direction to reduce loss
       - Repeats until convergence
    
    2. CONVERGENCE occurs when:
       - Additional iterations don't significantly reduce loss
       - Parameters stabilize around optimal values
       - The model has found the best fit for the data
    
    3. LOSS CURVE shows:
       - Steep decline at start (rapid learning)
       - Gradual decrease (fine-tuning)
       - Flattening (convergence)
    
    4. CONVEX LOSS SURFACE guarantees:
       - Only ONE minimum (global minimum)
       - Gradient descent will always find it
       - No risk of getting stuck in local minima
    
    5. LEARNING RATE controls:
       - How big steps we take in each iteration
       - Too small: slow convergence
       - Too large: may overshoot or diverge
       - Just right: smooth, efficient convergence
    
    Reference: https://developers.google.com/machine-learning/crash-course/
               linear-regression/gradient-descent
    """)
    
    print("Done!")


# Run main function when script is executed directly
if __name__ == "__main__":
    main()

