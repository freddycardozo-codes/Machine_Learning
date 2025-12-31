"""
NumPy Random Examples: uniform and normal distributions
========================================================
This script demonstrates the usage of np.random.uniform and np.random.normal
with practical examples for neural network and data science applications.

Author: NumPy Tutorial
"""

import numpy as np

# =============================================================================
# PART 1: np.random.uniform - Uniform Distribution
# =============================================================================
# Generates random numbers where every value in the range has equal probability.
# Syntax: np.random.uniform(low=0.0, high=1.0, size=None)

print("=" * 60)
print("   PART 1: np.random.uniform - Uniform Distribution")
print("=" * 60)
print()

# Set seed for reproducibility
np.random.seed(42)

# -----------------------------------------------------------------------------
# Example 1: Single random number between 0 and 1 (default)
# -----------------------------------------------------------------------------
single_value = np.random.uniform()
print("Example 1: Single value [0, 1)")
print(f"   Result: {single_value:.4f}")
print()

# -----------------------------------------------------------------------------
# Example 2: Single random number between custom range
# -----------------------------------------------------------------------------
value_1_to_10 = np.random.uniform(1, 10)
print("Example 2: Single value [1, 10)")
print(f"   Result: {value_1_to_10:.4f}")
print()

# -----------------------------------------------------------------------------
# Example 3: Array of 5 random numbers between 0 and 100
# -----------------------------------------------------------------------------
array_5 = np.random.uniform(0, 100, 5)
print("Example 3: Array of 5 values [0, 100)")
print(f"   Result: {array_5.round(2)}")
print()

# -----------------------------------------------------------------------------
# Example 4: 2D array (3 rows, 4 columns) between -1 and 1
# -----------------------------------------------------------------------------
array_2d = np.random.uniform(-1, 1, (3, 4))
print("Example 4: 2D array (3x4) [-1, 1)")
print(f"   Result:\n{array_2d.round(2)}")
print()

# -----------------------------------------------------------------------------
# Example 5: Generate random distances (1 to 50 km) - Neural Network Use Case
# -----------------------------------------------------------------------------
np.random.seed(42)
distances = np.random.uniform(1, 50, 10)
print("Example 5: Random distances for delivery prediction (km)")
print(f"   Result: {distances.round(1)}")
print()

# -----------------------------------------------------------------------------
# Example 6: Random prices between $9.99 and $99.99
# -----------------------------------------------------------------------------
prices = np.random.uniform(9.99, 99.99, 5)
print("Example 6: Random product prices ($)")
print(f"   Result: ${prices.round(2)}")
print()

# -----------------------------------------------------------------------------
# Example 7: Random RGB color values (0-255)
# -----------------------------------------------------------------------------
rgb_colors = np.random.uniform(0, 256, (3, 3)).astype(int)
print("Example 7: Random RGB colors (3 colors)")
print(f"   Result:\n{rgb_colors}")
print("   (Each row is one color: [R, G, B])")
print()

# -----------------------------------------------------------------------------
# Example 8: Random coordinates in a square region
# -----------------------------------------------------------------------------
x_coords = np.random.uniform(-10, 10, 5)
y_coords = np.random.uniform(-10, 10, 5)
print("Example 8: Random (x, y) coordinates in [-10, 10]")
print(f"   Points: {list(zip(x_coords.round(2), y_coords.round(2)))}")
print()


# =============================================================================
# PART 2: np.random.normal - Normal (Gaussian) Distribution
# =============================================================================
# Generates random numbers following a bell curve centered at mean with spread std.
# Syntax: np.random.normal(loc=0.0, scale=1.0, size=None)
#   loc = mean (center of distribution)
#   scale = standard deviation (spread of distribution)

print("=" * 60)
print("   PART 2: np.random.normal - Normal (Gaussian) Distribution")
print("=" * 60)
print()

# Set seed for reproducibility
np.random.seed(42)

# -----------------------------------------------------------------------------
# Example 1: Standard normal distribution (mean=0, std=1)
# -----------------------------------------------------------------------------
standard_normal = np.random.normal()
print("Example 1: Standard normal value (mean=0, std=1)")
print(f"   Result: {standard_normal:.4f}")
print()

# -----------------------------------------------------------------------------
# Example 2: Array of 5 values from standard normal
# -----------------------------------------------------------------------------
array_std_normal = np.random.normal(0, 1, 5)
print("Example 2: Array of 5 standard normal values")
print(f"   Result: {array_std_normal.round(4)}")
print()

# -----------------------------------------------------------------------------
# Example 3: Custom mean and std - heights of adults (mean=170cm, std=10cm)
# -----------------------------------------------------------------------------
heights = np.random.normal(170, 10, 10)
print("Example 3: Random heights (mean=170cm, std=10cm)")
print(f"   Result: {heights.round(1)}")
print()

# -----------------------------------------------------------------------------
# Example 4: 2D array with custom distribution
# -----------------------------------------------------------------------------
array_2d_normal = np.random.normal(100, 15, (3, 4))
print("Example 4: 2D array (3x4) (mean=100, std=15)")
print(f"   Result:\n{array_2d_normal.round(1)}")
print()

# -----------------------------------------------------------------------------
# Example 5: Adding noise to data - Neural Network Use Case
# This is exactly how noise is added in delivery_time_nn.py
# -----------------------------------------------------------------------------
np.random.seed(42)
base_values = np.array([10, 20, 30, 40, 50])
noise_level = 0.1  # 10% noise

# Noise proportional to values (heteroscedastic noise)
noise = np.random.normal(0, noise_level * base_values)
noisy_values = base_values + noise

print("Example 5: Adding proportional noise to data (Neural Network)")
print(f"   Base values:  {base_values}")
print(f"   Noise added:  {noise.round(2)}")
print(f"   Noisy values: {noisy_values.round(2)}")
print("   Note: Larger values get more noise (heteroscedasticity)")
print()

# -----------------------------------------------------------------------------
# Example 6: Simulating test scores (mean=75, std=12)
# -----------------------------------------------------------------------------
np.random.seed(42)
test_scores = np.random.normal(75, 12, 100)
# Clip to valid range [0, 100]
test_scores = np.clip(test_scores, 0, 100)

print("Example 6: Simulated test scores (mean=75, std=12)")
print(f"   Generated Mean: {test_scores.mean():.1f}")
print(f"   Generated Std:  {test_scores.std():.1f}")
print(f"   Sample scores:  {test_scores[:10].round(0).astype(int)}")
print()

# -----------------------------------------------------------------------------
# Example 7: Stock price simulation (daily returns)
# -----------------------------------------------------------------------------
np.random.seed(42)
initial_price = 100
mean_return = 0.001  # 0.1% daily return
volatility = 0.02    # 2% daily volatility
days = 30

daily_returns = np.random.normal(mean_return, volatility, days)
prices = initial_price * np.cumprod(1 + daily_returns)

print("Example 7: Stock price simulation")
print(f"   Initial price: ${initial_price}")
print(f"   Daily mean return: {mean_return*100:.1f}%")
print(f"   Daily volatility: {volatility*100:.1f}%")
print(f"   Days simulated: {days}")
print(f"   Final price: ${prices[-1]:.2f}")
print(f"   Total return: {(prices[-1]/initial_price - 1)*100:.1f}%")
print()

# -----------------------------------------------------------------------------
# Example 8: Generate data for linear regression with noise
# -----------------------------------------------------------------------------
np.random.seed(42)
n_samples = 100
X = np.linspace(0, 10, n_samples)
true_slope = 2.5
true_intercept = 5
noise_std = 2

y_true = true_slope * X + true_intercept
y_noisy = y_true + np.random.normal(0, noise_std, n_samples)

print("Example 8: Linear regression data with Gaussian noise")
print(f"   True equation: y = {true_slope}x + {true_intercept}")
print(f"   Noise std: {noise_std}")
print(f"   First 5 X values: {X[:5].round(2)}")
print(f"   First 5 y (true):  {y_true[:5].round(2)}")
print(f"   First 5 y (noisy): {y_noisy[:5].round(2)}")
print()


# =============================================================================
# PART 3: Comparison - Uniform vs Normal
# =============================================================================

print("=" * 60)
print("   PART 3: Comparison - Uniform vs Normal")
print("=" * 60)
print()

np.random.seed(42)
n = 10000

# Generate samples
uniform_samples = np.random.uniform(0, 10, n)
normal_samples = np.random.normal(5, 1.5, n)  # mean=5, std=1.5

print("UNIFORM DISTRIBUTION [0, 10)")
print("-" * 40)
print(f"   Min:  {uniform_samples.min():.2f}")
print(f"   Max:  {uniform_samples.max():.2f}")
print(f"   Mean: {uniform_samples.mean():.2f} (expected: 5.0)")
print(f"   Std:  {uniform_samples.std():.2f} (expected: 2.89)")
print()

print("NORMAL DISTRIBUTION (mean=5, std=1.5)")
print("-" * 40)
print(f"   Min:  {normal_samples.min():.2f}")
print(f"   Max:  {normal_samples.max():.2f}")
print(f"   Mean: {normal_samples.mean():.2f} (expected: 5.0)")
print(f"   Std:  {normal_samples.std():.2f} (expected: 1.5)")
print()

# Distribution of values
print("VALUE DISTRIBUTION COMPARISON")
print("-" * 40)
print()
print("Uniform - values spread EVENLY across range:")
for i in range(0, 10, 2):
    count = np.sum((uniform_samples >= i) & (uniform_samples < i+2))
    bar = "#" * int(count / 100)
    print(f"   [{i:2d}-{i+2:2d}): {count:4d} ({count/n*100:5.1f}%) {bar}")

print()
print("Normal - values CONCENTRATED around mean (5):")
for i in range(0, 10, 2):
    count = np.sum((normal_samples >= i) & (normal_samples < i+2))
    bar = "#" * int(count / 100)
    print(f"   [{i:2d}-{i+2:2d}): {count:4d} ({count/n*100:5.1f}%) {bar}")
print()


# =============================================================================
# PART 4: Quick Reference Summary
# =============================================================================

print("=" * 60)
print("   PART 4: Quick Reference Summary")
print("=" * 60)
print()
print("np.random.uniform(low, high, size)")
print("-" * 40)
print("   • Every value in [low, high) has EQUAL probability")
print("   • Use for: random positions, IDs, sampling from range")
print("   • Default: low=0.0, high=1.0")
print()
print("np.random.normal(mean, std, size)")
print("-" * 40)
print("   • Values follow a BELL CURVE centered at mean")
print("   • Most values within mean ± 2*std (95%)")
print("   • Use for: realistic noise, natural measurements")
print("   • Default: mean=0.0, std=1.0 (standard normal)")
print()
print("Common Use Cases in Neural Networks:")
print("-" * 40)
print("   • uniform: Random input data, initial exploration")
print("   • normal:  Weight initialization, adding noise to data")
print()

print("=" * 60)
print("   Done!")
print("=" * 60)

