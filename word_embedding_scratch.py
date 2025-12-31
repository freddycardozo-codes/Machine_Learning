"""
Word Embedding Model from Scratch
==================================
A Word2Vec-style implementation using only NumPy to create word embeddings
from a paragraph of text.

This implements the Skip-gram model:
- Given a target word, predict surrounding context words
- The hidden layer weights become the word embeddings

Key Concepts:
- Word embeddings map words to dense vectors (e.g., 50-300 dimensions)
- Similar words have similar vectors (close in vector space)
- Captures semantic relationships (king - man + woman ≈ queen)
- Alternative to one-hot encoding which is sparse and doesn't capture similarity

Skip-gram vs CBOW:
- Skip-gram: Given target word, predict context words (used here)
- CBOW: Given context words, predict target word

Author: Neural Network Tutorial
"""

# Import NumPy for numerical operations
# NumPy provides efficient array operations essential for neural network computations
import numpy as np

# Import re (regular expressions) for text preprocessing
# Used to remove punctuation and clean text data
import re

# Import Counter from collections for counting word frequencies
# Counter is a dictionary subclass for counting hashable objects
from collections import Counter

# Try to import matplotlib for visualization
# We wrap in try/except to make script work without matplotlib installed
try:
    # matplotlib.pyplot provides plotting functions similar to MATLAB
    import matplotlib.pyplot as plt
    # Set flag indicating matplotlib is available
    HAS_MATPLOTLIB = True
except ImportError:
    # If matplotlib is not installed, set flag to False
    HAS_MATPLOTLIB = False
    # Inform user that visualizations will be skipped
    print("Note: matplotlib not available. Visualization will be skipped.")


class WordEmbedding:
    """
    Word2Vec Skip-gram model from scratch.
    
    Architecture:
        Input Layer:  V neurons (one-hot encoded word, V = vocabulary size)
        Hidden Layer: E neurons (embedding dimension)
        Output Layer: V neurons (softmax over vocabulary)
    
    The key insight: The hidden layer weights (W1) become the word embeddings!
    When we look up a word, we're just selecting a row from W1.
    
    Training objective: Given a target word, predict words that appear nearby.
    """
    
    def __init__(self, vocab_size, embedding_dim=50, learning_rate=0.01):
        """
        Initialize the embedding model with random weights.
        
        Args:
            vocab_size: Size of vocabulary (V) - number of unique words
            embedding_dim: Dimension of word vectors (E) - typically 50-300
            learning_rate: Learning rate for gradient descent optimization
        """
        # Store vocabulary size as instance variable
        # This determines the size of input/output layers
        self.vocab_size = vocab_size
        
        # Store embedding dimension as instance variable
        # This is the size of the dense vector each word will be mapped to
        # Smaller = faster training, larger = more semantic capacity
        self.embedding_dim = embedding_dim
        
        # Store learning rate for gradient descent
        # Controls how much weights are adjusted each training step
        self.learning_rate = learning_rate
        
        # ==================== WEIGHT INITIALIZATION ====================
        
        # Initialize W1: Input -> Hidden layer weights
        # Shape: (vocab_size, embedding_dim) = (V, E)
        # IMPORTANT: These weights ARE the word embeddings!
        # Each row i of W1 is the embedding vector for word i
        # np.random.randn generates samples from standard normal distribution
        # Multiply by 0.01 to start with small random values (helps training)
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Initialize W2: Hidden -> Output layer weights
        # Shape: (embedding_dim, vocab_size) = (E, V)
        # These weights transform the hidden representation to output probabilities
        # Also called "context embeddings" in some literature
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01
    
    def softmax(self, x):
        """
        Softmax function - converts scores to probability distribution.
        
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        
        Properties:
        - Output values are between 0 and 1
        - All outputs sum to 1 (valid probability distribution)
        - Preserves relative ordering (larger input = larger output)
        
        Args:
            x: Input array of scores/logits, shape (V,)
            
        Returns:
            Probability distribution over vocabulary, shape (V,)
            Each value represents probability of that word being the context word
        """
        # Subtract max value for numerical stability
        # This prevents overflow when computing exp of large numbers
        # Mathematically: softmax(x - c) = softmax(x) for any constant c
        # np.max(x) finds the maximum value in the array
        # np.exp computes e^x element-wise
        exp_x = np.exp(x - np.max(x))
        
        # Divide by sum to normalize to probability distribution
        # np.sum(exp_x) adds up all the exponential values
        # Result: each element is exp(x_i) / sum(exp(x_j))
        return exp_x / np.sum(exp_x)
    
    def forward(self, x_one_hot):
        """
        Forward propagation through the network.
        
        Architecture:
        Input (one-hot) -> W1 -> Hidden (embedding) -> W2 -> Output (softmax)
        
        The forward pass computes:
        1. hidden = x_one_hot @ W1  (lookup embedding)
        2. output = softmax(hidden @ W2)  (predict context word)
        
        Args:
            x_one_hot: One-hot encoded input word, shape (V,)
                       e.g., [0, 0, 1, 0, 0, ...] for word at index 2
            
        Returns:
            hidden: Hidden layer activations (the embedding), shape (E,)
            output: Softmax probabilities over vocabulary, shape (V,)
        """
        # ==================== HIDDEN LAYER (EMBEDDING LOOKUP) ====================
        
        # Compute hidden layer: matrix multiplication of one-hot with W1
        # Since x_one_hot is one-hot (only one 1, rest zeros):
        # - This is equivalent to selecting the row of W1 for the input word
        # - If x_one_hot[i] = 1, then hidden = W1[i, :] (row i of W1)
        # Mathematical operation: hidden = x_one_hot @ W1
        # np.dot performs matrix/vector multiplication
        # Shape: (V,) @ (V, E) = (E,) - the embedding vector!
        hidden = np.dot(x_one_hot, self.W1)  # Shape: (E,)
        
        # ==================== OUTPUT LAYER (CONTEXT PREDICTION) ====================
        
        # Compute output scores: transform embedding to vocabulary-sized output
        # This computes a score for each word in vocabulary
        # Higher score = more likely to be a context word
        # Mathematical operation: output_linear = hidden @ W2
        # Shape: (E,) @ (E, V) = (V,) - score for each word
        output_linear = np.dot(hidden, self.W2)  # Shape: (V,)
        
        # Apply softmax to convert scores to probabilities
        # Now each value represents P(word_j | word_i)
        # i.e., probability that word_j appears near word_i
        output = self.softmax(output_linear)  # Shape: (V,)
        
        # Return both hidden (embedding) and output (predictions)
        # We need hidden for backpropagation
        return hidden, output
    
    def backward(self, x_one_hot, hidden, output, y_one_hot):
        """
        Backpropagation to compute gradients and update weights.
        
        Loss function: Cross-entropy loss
        L = -log(output[correct_word_idx])
        
        For softmax + cross-entropy, the gradient simplifies beautifully:
        dL/d(output_linear) = output - y_one_hot
        
        Args:
            x_one_hot: One-hot encoded input word, shape (V,)
            hidden: Hidden layer output from forward pass, shape (E,)
            output: Softmax output from forward pass, shape (V,)
            y_one_hot: One-hot encoded target (context) word, shape (V,)
        """
        # ==================== OUTPUT LAYER GRADIENTS ====================
        
        # Compute output layer error (gradient of loss w.r.t. pre-softmax scores)
        # For cross-entropy loss with softmax, this simplifies to:
        # dL/dz2 = output - y_one_hot = predicted - actual
        # Shape: (V,) - (V,) = (V,)
        # This is the error signal: positive where we over-predicted,
        # negative where we under-predicted
        output_error = output - y_one_hot  # Shape: (V,)
        
        # Compute gradient for W2
        # dL/dW2 = hidden^T @ output_error
        # np.outer computes outer product: (E,) x (V,) = (E, V)
        # Each element dW2[i,j] = hidden[i] * output_error[j]
        # This shows how much to adjust each weight to reduce error
        dW2 = np.outer(hidden, output_error)
        
        # ==================== HIDDEN LAYER GRADIENTS ====================
        
        # Propagate error back to hidden layer
        # dL/dhidden = W2 @ output_error
        # This tells us how each hidden unit contributed to the error
        # np.dot: (E, V) @ (V,) = (E,)
        hidden_error = np.dot(self.W2, output_error)  # Shape: (E,)
        
        # Compute gradient for W1
        # dL/dW1 = x_one_hot^T @ hidden_error
        # np.outer computes outer product: (V,) x (E,) = (V, E)
        # Since x_one_hot is one-hot, only one row of dW1 is non-zero
        # This is the row corresponding to the input word
        dW1 = np.outer(x_one_hot, hidden_error)
        
        # ==================== WEIGHT UPDATES (GRADIENT DESCENT) ====================
        
        # Update W1 using gradient descent
        # W1_new = W1_old - learning_rate * gradient
        # The negative sign moves weights in direction that reduces loss
        # Only the row for the input word actually changes (due to one-hot)
        self.W1 -= self.learning_rate * dW1
        
        # Update W2 using gradient descent
        # All columns can change since hidden is dense
        self.W2 -= self.learning_rate * dW2
    
    def train_step(self, x_idx, y_idx):
        """
        Single training step for one (input word, context word) pair.
        
        This is the core training function that:
        1. Creates one-hot encodings
        2. Runs forward pass
        3. Computes loss
        4. Runs backward pass (updates weights)
        
        Args:
            x_idx: Index of input (target) word in vocabulary
            y_idx: Index of context word in vocabulary
            
        Returns:
            loss: Cross-entropy loss for this training pair
        """
        # ==================== CREATE ONE-HOT ENCODINGS ====================
        
        # Create one-hot encoding for input word
        # np.zeros creates array of zeros with specified shape
        # Shape: (vocab_size,) = (V,)
        x_one_hot = np.zeros(self.vocab_size)
        
        # Set the position corresponding to input word to 1
        # All other positions remain 0
        # Result: [0, 0, ..., 1, ..., 0, 0] with 1 at position x_idx
        x_one_hot[x_idx] = 1
        
        # Create one-hot encoding for target (context) word
        # Same process as input word
        y_one_hot = np.zeros(self.vocab_size)
        
        # Set the position corresponding to context word to 1
        y_one_hot[y_idx] = 1
        
        # ==================== FORWARD PASS ====================
        
        # Run forward propagation to get predictions
        # hidden: the embedding of the input word, shape (E,)
        # output: predicted probabilities for each word, shape (V,)
        hidden, output = self.forward(x_one_hot)
        
        # ==================== COMPUTE LOSS ====================
        
        # Compute cross-entropy loss
        # Loss = -log(predicted probability of correct context word)
        # Lower loss = model predicted the correct word with high probability
        # output[y_idx] is the predicted probability for the true context word
        # Add 1e-10 (small epsilon) to prevent log(0) which is undefined
        # np.log computes natural logarithm
        loss = -np.log(output[y_idx] + 1e-10)
        
        # ==================== BACKWARD PASS ====================
        
        # Run backpropagation to update weights
        # This computes gradients and adjusts W1 and W2
        self.backward(x_one_hot, hidden, output, y_one_hot)
        
        # Return the loss for monitoring training progress
        return loss
    
    def get_embedding(self, word_idx):
        """
        Get the embedding vector for a specific word.
        
        The embedding is simply the row of W1 corresponding to that word.
        This is what makes Word2Vec efficient: embedding lookup is just
        indexing into a matrix.
        
        Args:
            word_idx: Index of the word in vocabulary
            
        Returns:
            Embedding vector of shape (E,) - dense representation of the word
        """
        # Return the row of W1 for this word
        # W1[word_idx] selects row word_idx from the matrix
        # Shape: (E,) - a vector of embedding_dim dimensions
        return self.W1[word_idx]
    
    def get_all_embeddings(self):
        """
        Get all word embeddings (the entire W1 matrix).
        
        Returns:
            All embeddings, shape (V, E)
            Row i is the embedding for word i
        """
        # Return the entire W1 matrix
        # Each row is one word's embedding vector
        return self.W1


class TextPreprocessor:
    """
    Preprocess text for word embedding training.
    
    This class handles:
    1. Tokenization: splitting text into words
    2. Vocabulary building: mapping words to indices
    3. Training data generation: creating (target, context) pairs
    """
    
    def __init__(self):
        """Initialize the preprocessor with empty mappings."""
        # Dictionary mapping words to their integer indices
        # e.g., {"cat": 0, "dog": 1, "bird": 2, ...}
        self.word2idx = {}
        
        # Dictionary mapping indices back to words (reverse of word2idx)
        # e.g., {0: "cat", 1: "dog", 2: "bird", ...}
        self.idx2word = {}
        
        # Size of vocabulary (number of unique words)
        self.vocab_size = 0
        
        # Counter object to track word frequencies
        # Counter is a dict subclass that counts occurrences
        self.word_counts = Counter()
    
    def tokenize(self, text):
        """
        Convert text to lowercase tokens (words).
        
        Steps:
        1. Convert to lowercase (case-insensitive)
        2. Remove punctuation and special characters
        3. Split into individual words
        
        Args:
            text: Input text string (paragraph, document, etc.)
            
        Returns:
            List of word tokens (strings)
        """
        # Convert entire text to lowercase
        # This ensures "Machine" and "machine" are treated as same word
        # str.lower() returns lowercase version of string
        text = text.lower()
        
        # Remove punctuation and special characters using regex
        # re.sub(pattern, replacement, string) substitutes matches
        # Pattern [^a-z0-9\s] matches anything that is NOT:
        #   a-z: lowercase letters
        #   0-9: digits
        #   \s: whitespace
        # Replace all such characters with empty string ''
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Split text into words by whitespace
        # str.split() with no argument splits on any whitespace
        # Returns list of words: ["machine", "learning", "is", ...]
        words = text.split()
        
        # Return the list of word tokens
        return words
    
    def build_vocabulary(self, words, min_count=1):
        """
        Build vocabulary from list of words.
        
        Creates mappings between words and integer indices.
        Only includes words that appear at least min_count times.
        
        Args:
            words: List of word tokens
            min_count: Minimum frequency to include word in vocabulary
                      (helps filter rare words that can't be learned well)
        """
        # Count frequency of each word
        # Counter(words) creates dict-like object with word counts
        # e.g., Counter({"machine": 10, "learning": 8, "the": 15, ...})
        self.word_counts = Counter(words)
        
        # Build vocabulary with words meeting minimum count threshold
        # Initialize index counter
        idx = 0
        
        # Iterate through all words and their counts
        # .items() returns (word, count) pairs
        for word, count in self.word_counts.items():
            # Only include word if it appears at least min_count times
            # This filters out rare words that don't have enough context
            if count >= min_count:
                # Add word to word2idx mapping
                # Assigns next available integer index to this word
                self.word2idx[word] = idx
                
                # Add reverse mapping (index to word)
                # Useful for converting model output back to words
                self.idx2word[idx] = word
                
                # Increment index for next word
                idx += 1
        
        # Store total vocabulary size
        # len() returns number of items in dictionary
        self.vocab_size = len(self.word2idx)
        
        # Print vocabulary size for user feedback
        print(f"   Vocabulary size: {self.vocab_size} words")
    
    def generate_training_data(self, words, window_size=2):
        """
        Generate (target, context) pairs for Skip-gram training.
        
        For each word in the text, we create training pairs with words
        that appear within a window of +/- window_size positions.
        
        Example: "the cat sat on the mat" with window_size=2
        For target word "sat" at position 2:
            Context window: positions 0,1,3,4 → words: "the", "cat", "on", "the"
            Training pairs: (sat, the), (sat, cat), (sat, on), (sat, the)
        
        The model learns: if I see "sat", predict "the", "cat", "on" nearby
        
        Args:
            words: List of word tokens from the text
            window_size: Number of words on each side to consider as context
            
        Returns:
            List of (target_idx, context_idx) tuples for training
        """
        # Initialize empty list to store training pairs
        training_data = []
        
        # Iterate through each word in the text with its position
        # enumerate returns (index, value) pairs
        for i, word in enumerate(words):
            # Skip words that aren't in our vocabulary
            # (they didn't meet the min_count threshold)
            if word not in self.word2idx:
                continue
            
            # Get the vocabulary index of the target (center) word
            # This is the word we're trying to learn embeddings for
            target_idx = self.word2idx[word]
            
            # Define the context window boundaries
            # start: leftmost position (but not before beginning of text)
            # max(0, i - window_size) ensures we don't go negative
            start = max(0, i - window_size)
            
            # end: rightmost position + 1 (but not past end of text)
            # min(len(words), i + window_size + 1) ensures we don't exceed text length
            # +1 because range is exclusive of end
            end = min(len(words), i + window_size + 1)
            
            # Iterate through all positions in the context window
            for j in range(start, end):
                # Skip if j == i (don't pair word with itself)
                # Also skip if context word not in vocabulary
                if j != i and words[j] in self.word2idx:
                    # Get vocabulary index of context word
                    context_idx = self.word2idx[words[j]]
                    
                    # Add (target, context) pair to training data
                    # The model will learn: when target appears, context is likely nearby
                    training_data.append((target_idx, context_idx))
        
        # Print number of training pairs generated
        print(f"   Training pairs: {len(training_data)}")
        
        # Return the list of training pairs
        return training_data


def cosine_similarity(v1, v2):
    """
    Compute cosine similarity between two vectors.
    
    Cosine similarity measures the angle between two vectors:
    - 1: vectors point in same direction (most similar)
    - 0: vectors are perpendicular (unrelated)
    - -1: vectors point in opposite directions (most dissimilar)
    
    Formula: cos(θ) = (v1 · v2) / (||v1|| * ||v2||)
    
    Where:
    - v1 · v2 is the dot product
    - ||v|| is the Euclidean norm (length) of vector v
    
    Args:
        v1: First vector, shape (E,)
        v2: Second vector, shape (E,)
        
    Returns:
        Cosine similarity value between -1 and 1
    """
    # Compute dot product of the two vectors
    # np.dot for 1D arrays computes: sum(v1[i] * v2[i]) for all i
    dot_product = np.dot(v1, v2)
    
    # Compute Euclidean norm (length) of each vector
    # np.linalg.norm computes: sqrt(sum(v[i]^2)) for all i
    # This is the L2 norm or "magnitude" of the vector
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Handle edge case: if either vector has zero length
    # Can't divide by zero, so return 0 similarity
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    # Compute and return cosine similarity
    # Divide dot product by product of norms
    return dot_product / (norm_v1 * norm_v2)


def find_similar_words(model, preprocessor, word, top_n=5):
    """
    Find the most similar words to a given word based on embedding similarity.
    
    This demonstrates that learned embeddings capture semantic relationships:
    words that appear in similar contexts have similar embedding vectors.
    
    Args:
        model: Trained WordEmbedding model with learned embeddings
        preprocessor: TextPreprocessor with vocabulary mappings
        word: Target word to find similar words for
        top_n: Number of most similar words to return
        
    Returns:
        List of (word, similarity_score) tuples, sorted by similarity
    """
    # Check if the word exists in our vocabulary
    # If not, we can't find its embedding
    if word not in preprocessor.word2idx:
        print(f"   '{word}' not in vocabulary")
        # Return empty list if word not found
        return []
    
    # Get the vocabulary index of the target word
    word_idx = preprocessor.word2idx[word]
    
    # Get the embedding vector for the target word
    # This is a dense vector of shape (E,)
    word_vec = model.get_embedding(word_idx)
    
    # Initialize list to store (word, similarity) pairs
    similarities = []
    
    # Compare target word with every other word in vocabulary
    for idx in range(preprocessor.vocab_size):
        # Skip comparing word with itself (similarity would be 1.0)
        if idx != word_idx:
            # Get embedding of the other word
            other_vec = model.get_embedding(idx)
            
            # Compute cosine similarity between the two embeddings
            sim = cosine_similarity(word_vec, other_vec)
            
            # Store the word (not index) and its similarity score
            # preprocessor.idx2word converts index back to word string
            similarities.append((preprocessor.idx2word[idx], sim))
    
    # Sort by similarity score in descending order (most similar first)
    # key=lambda x: x[1] sorts by the second element (similarity)
    # reverse=True means highest similarity first
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return only the top_n most similar words
    # [:top_n] slices the list to get first top_n elements
    return similarities[:top_n]


def plot_embeddings_2d(model, preprocessor, words_to_plot=None):
    """
    Visualize word embeddings in 2D using PCA-like dimensionality reduction.
    
    Since embeddings are high-dimensional (e.g., 50D), we project them to 2D
    for visualization. We use Principal Component Analysis (PCA) which finds
    the 2 directions of maximum variance in the data.
    
    Args:
        model: Trained WordEmbedding model
        preprocessor: TextPreprocessor with vocabulary
        words_to_plot: Optional list of specific words to plot
    """
    # Check if matplotlib is available
    if not HAS_MATPLOTLIB:
        print("   (Skipping plot - matplotlib not available)")
        return  # Exit function early
    
    # ==================== GET EMBEDDINGS ====================
    
    # Get all word embeddings from the model
    # Shape: (vocab_size, embedding_dim) = (V, E)
    embeddings = model.get_all_embeddings()
    
    # ==================== PCA: DIMENSIONALITY REDUCTION ====================
    
    # Step 1: Center the data (subtract mean)
    # np.mean(embeddings, axis=0) computes mean of each column
    # Centering is required for PCA
    # Shape: (V, E) - (E,) = (V, E) via broadcasting
    embeddings_centered = embeddings - np.mean(embeddings, axis=0)
    
    # Step 2: Compute covariance matrix
    # Covariance measures how dimensions vary together
    # np.cov expects features as rows, so we transpose (.T)
    # Shape: (E, E) - square matrix
    cov = np.cov(embeddings_centered.T)
    
    # Step 3: Compute eigenvalues and eigenvectors of covariance matrix
    # Eigenvectors point in directions of maximum variance
    # Eigenvalues indicate how much variance each direction captures
    # np.linalg.eig returns (eigenvalues array, eigenvectors matrix)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Step 4: Sort eigenvectors by eigenvalues (descending)
    # We want the directions with most variance first
    # np.argsort returns indices that would sort the array
    # [::-1] reverses to get descending order
    idx = np.argsort(eigenvalues)[::-1]
    
    # Reorder eigenvectors by sorted indices
    # eigenvectors[:, idx] reorders columns
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Project to 2D using top 2 eigenvectors
    # These capture the most variance (best 2D representation)
    # eigenvectors[:, :2] selects first 2 columns
    # .real extracts real part (eigenvalues can be complex)
    projection_matrix = eigenvectors[:, :2].real
    
    # Project all embeddings to 2D
    # Matrix multiplication: (V, E) @ (E, 2) = (V, 2)
    # Each row is now a 2D point for visualization
    embeddings_2d = np.dot(embeddings_centered, projection_matrix)
    
    # ==================== SELECT WORDS TO PLOT ====================
    
    # If no specific words provided, choose automatically
    if words_to_plot is None:
        # If vocabulary is small, plot all words
        if preprocessor.vocab_size <= 50:
            # list() converts dictionary keys to a list
            words_to_plot = list(preprocessor.word2idx.keys())
        else:
            # For large vocabulary, plot most frequent words
            # .most_common(30) returns 30 most frequent (word, count) pairs
            # List comprehension extracts just the words
            words_to_plot = [word for word, _ in preprocessor.word_counts.most_common(30)]
    
    # ==================== CREATE PLOT ====================
    
    # Create a new figure with specified size
    # figsize=(width, height) in inches
    plt.figure(figsize=(12, 10))
    
    # Plot each word
    for word in words_to_plot:
        # Check if word is in vocabulary
        if word in preprocessor.word2idx:
            # Get vocabulary index
            idx = preprocessor.word2idx[word]
            
            # Get 2D coordinates for this word
            # embeddings_2d[idx] is the 2D point for this word
            x, y = embeddings_2d[idx]
            
            # Plot as a scatter point
            # marker='o': circle marker
            # s=100: size of marker
            # alpha=0.7: 70% opacity
            plt.scatter(x, y, marker='o', s=100, alpha=0.7)
            
            # Add word label next to the point
            # plt.annotate adds text at specified position
            # fontsize=10: text size
            # ha='center': horizontal alignment
            # va='bottom': vertical alignment (text above point)
            plt.annotate(word, (x, y), fontsize=10, ha='center', va='bottom')
    
    # Add title and axis labels
    plt.title('Word Embeddings Visualization (2D Projection)', fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    
    # Add grid for easier reading
    # alpha=0.3: 30% opacity (subtle grid)
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label clipping
    plt.tight_layout()
    
    # Save figure to file
    # dpi=150: dots per inch (resolution)
    # bbox_inches='tight': minimize whitespace
    plt.savefig('word_embeddings_2d.png', dpi=150, bbox_inches='tight')
    
    # Display the plot
    plt.show()


def plot_training_loss(losses):
    """
    Plot training loss over epochs to visualize learning progress.
    
    A decreasing loss curve indicates the model is learning.
    
    Args:
        losses: List of average loss values, one per epoch
    """
    # Check if matplotlib is available
    if not HAS_MATPLOTLIB:
        return  # Exit early if no plotting available
    
    # Create figure with specified size
    plt.figure(figsize=(10, 5))
    
    # Plot loss values
    # losses is a list, x-axis will be indices (epochs)
    # color='#3498db': blue color
    # linewidth=1.5: line thickness
    plt.plot(losses, color='#3498db', linewidth=1.5)
    
    # Add axis labels
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    
    # Add title
    plt.title('Word Embedding Training Loss', fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to file
    plt.savefig('embedding_training_loss.png', dpi=150, bbox_inches='tight')
    
    # Display
    plt.show()


def main():
    """
    Main function that orchestrates the entire word embedding training pipeline.
    
    Steps:
    1. Define sample text
    2. Preprocess text (tokenize, build vocabulary)
    3. Generate training data (target, context pairs)
    4. Create and train the embedding model
    5. Evaluate by finding similar words
    6. Visualize the learned embeddings
    """
    # Print header
    print("=" * 60)
    print("   Word Embedding Model from Scratch")
    print("=" * 60)
    print()
    
    # ==================== SAMPLE TEXT ====================
    
    # Define a sample paragraph for training
    # This text has repeated concepts about ML/AI so the model can learn relationships
    # In real applications, you'd use much larger text corpora
    paragraph = """
    Machine learning is a subset of artificial intelligence. 
    Deep learning is a subset of machine learning. 
    Neural networks are the foundation of deep learning.
    Artificial intelligence includes machine learning and deep learning.
    Data science uses machine learning for predictions.
    Machine learning algorithms learn from data.
    Neural networks consist of layers of neurons.
    Deep learning uses multiple layers in neural networks.
    Artificial intelligence is transforming technology.
    Data is essential for machine learning models.
    Training neural networks requires large amounts of data.
    Deep learning models can recognize patterns in data.
    Machine learning models improve with more training data.
    Artificial intelligence applications include natural language processing.
    Natural language processing is part of artificial intelligence.
    Computer vision uses deep learning for image recognition.
    Image recognition is a task for neural networks.
    Machine learning helps in making predictions from data.
    The future of technology depends on artificial intelligence.
    Neural networks are inspired by the human brain.
    """
    
    # Print sample of the text
    print("Sample Text:")
    print("-" * 40)
    # [:200] shows first 200 characters
    print(paragraph[:200] + "...")
    print("-" * 40)
    print()
    
    # ==================== TEXT PREPROCESSING ====================
    
    print("Preprocessing text...")
    
    # Create preprocessor instance
    preprocessor = TextPreprocessor()
    
    # Tokenize the paragraph into words
    # Returns list like ["machine", "learning", "is", ...]
    words = preprocessor.tokenize(paragraph)
    
    # Print total word count
    print(f"   Total words: {len(words)}")
    
    # Build vocabulary with minimum count threshold
    # min_count=2 filters out words appearing only once
    preprocessor.build_vocabulary(words, min_count=2)
    
    # ==================== GENERATE TRAINING DATA ====================
    
    print("Generating training pairs (Skip-gram)...")
    
    # Generate (target, context) pairs for training
    # window_size=2 means we look 2 words left and 2 words right
    training_data = preprocessor.generate_training_data(words, window_size=2)
    print()
    
    # ==================== CREATE MODEL ====================
    
    print("Creating Word Embedding Model...")
    
    # Set embedding dimension
    # Small (20) for this example; typically 50-300 in practice
    embedding_dim = 20
    
    # Create the word embedding model
    # vocab_size determines input/output layer sizes
    # embedding_dim is the hidden layer size (our embedding vectors)
    # learning_rate controls training speed
    model = WordEmbedding(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=embedding_dim,
        learning_rate=0.05
    )
    
    print(f"   Embedding dimension: {embedding_dim}")
    print()
    
    # ==================== TRAINING ====================
    
    print("Training the model...")
    print("-" * 40)
    
    # Number of complete passes through training data
    epochs = 100
    
    # List to store average loss per epoch
    losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle training data each epoch
        # This helps prevent the model from learning order-dependent patterns
        # np.random.shuffle modifies the list in-place
        np.random.shuffle(training_data)
        
        # Accumulate loss for this epoch
        epoch_loss = 0
        
        # Train on each (target, context) pair
        for target_idx, context_idx in training_data:
            # Perform one training step
            # Returns cross-entropy loss for this pair
            loss = model.train_step(target_idx, context_idx)
            
            # Add to total loss for this epoch
            epoch_loss += loss
        
        # Compute average loss for this epoch
        # Divide by number of training pairs
        avg_loss = epoch_loss / len(training_data)
        
        # Store for plotting
        losses.append(avg_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print("-" * 40)
    print()
    
    # ==================== SHOW VOCABULARY ====================
    
    print("=" * 60)
    print("   Vocabulary")
    print("=" * 60)
    print()
    
    # Print all words in vocabulary (sorted alphabetically)
    print("Words in vocabulary:")
    # sorted() returns sorted list of dictionary keys
    # ", ".join() combines list into comma-separated string
    print(", ".join(sorted(preprocessor.word2idx.keys())))
    print()
    
    # ==================== FIND SIMILAR WORDS ====================
    
    print("=" * 60)
    print("   Similar Words (Cosine Similarity)")
    print("=" * 60)
    print()
    
    # Test words to find similar words for
    test_words = ['learning', 'neural', 'data', 'intelligence', 'deep']
    
    # For each test word, find and print similar words
    for word in test_words:
        # Check if word is in vocabulary
        if word in preprocessor.word2idx:
            print(f"Words similar to '{word}':")
            
            # Find top 5 similar words
            similar = find_similar_words(model, preprocessor, word, top_n=5)
            
            # Print each similar word and its score
            for similar_word, score in similar:
                # :.4f formats float with 4 decimal places
                print(f"   {similar_word}: {score:.4f}")
            print()
    
    # ==================== SHOW SAMPLE EMBEDDINGS ====================
    
    print("=" * 60)
    print("   Sample Word Embeddings")
    print("=" * 60)
    print()
    
    # Words to show embeddings for
    sample_words = ['machine', 'learning', 'neural', 'networks', 'data']
    
    for word in sample_words:
        # Check if word is in vocabulary
        if word in preprocessor.word2idx:
            # Get word's index
            idx = preprocessor.word2idx[word]
            
            # Get the embedding vector
            embedding = model.get_embedding(idx)
            
            # Print first 10 dimensions
            # [:10] slices first 10 elements
            # .round(4) rounds to 4 decimal places
            print(f"'{word}' embedding (first 10 dims):")
            print(f"   {embedding[:10].round(4)}")
            print()
    
    # ==================== VISUALIZATIONS ====================
    
    # Only create visualizations if matplotlib is available
    if HAS_MATPLOTLIB:
        print("Generating visualizations...")
        
        # Plot training loss curve
        plot_training_loss(losses)
        print("   Saved 'embedding_training_loss.png'")
        
        # Plot 2D embedding visualization
        plot_embeddings_2d(model, preprocessor)
        print("   Saved 'word_embeddings_2d.png'")
    
    # ==================== EXPLANATION ====================
    
    # Print explanation of how the model works
    print()
    print("=" * 60)
    print("   How Word Embeddings Work")
    print("=" * 60)
    print("""
    Skip-gram Model (Word2Vec):
    ---------------------------
    1. For each word in the text, we try to predict surrounding words
    2. The neural network has 3 layers:
       - Input:  One-hot encoded word (size = vocabulary)
       - Hidden: Dense embedding (size = embedding_dim)
       - Output: Softmax probabilities (size = vocabulary)
    
    3. The magic happens in the hidden layer:
       - Each word is mapped to a dense vector
       - Words that appear in similar contexts get similar vectors
       - The hidden layer weights (W1) ARE the word embeddings!
    
    Why this works:
    ---------------
    - Words that appear in similar contexts have similar meanings
    - "king" and "queen" appear near "royal", "crown", "throne"
    - So their embedding vectors become similar
    
    Mathematical Intuition:
    -----------------------
    - One-hot input: [0, 0, 1, 0, 0, ...] selects one row of W1
    - That row IS the embedding for that word
    - Training adjusts W1 so similar words have similar rows
    """)
    
    print()
    print("Done!")


# Standard Python idiom: only run main() when script is executed directly
# __name__ equals "__main__" when running as a script
# __name__ equals the module name when imported
if __name__ == "__main__":
    # Execute the main function
    main()
