import os
import itertools
import numpy as np
from typing import List, Tuple, Dict

class KneserNeyNgramModel:
    def __init__(self, vocab_size: int, seq_len: int, discount: float = 0.1):
        """
        Initialize the Kneser-Ney N-gram model.

        Args:
            vocab_size (int): Size of the vocabulary
            seq_len (int): Length of the n-gram sequence
            discount (float): Discount parameter for Kneser-Ney smoothing
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.discount = discount
        self.counts = np.zeros((vocab_size,) * seq_len, dtype=np.uint32)  # Higher-order counts
        self.context_counts = np.zeros((vocab_size,) * (seq_len - 1), dtype=np.uint32)  # Context counts
        self.continuation_counts = np.zeros(vocab_size, dtype=np.uint32)  # Continuation counts
        self.total_count = 0

    def train(self, tape: List[int]) -> None:
        """
        Train the model with a given n-gram sequence.

        Args:
            tape (List[int]): Sequence of tokens representing the n-gram
        """
        if len(tape) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {len(tape)}")
        self.counts[tuple(tape)] += 1
        self.context_counts[tuple(tape[:-1])] += 1
        self.continuation_counts[tape[-1]] += 1
        self.total_count += 1

    def get_probs(self, context: List[int]) -> np.ndarray:
        """
        Get the probability distribution for the next token given a context using Kneser-Ney smoothing.

        Args:
            context (List[int]): Context sequence of tokens

        Returns:
            np.ndarray: Probability distribution for the next token
        """
        if len(context) != self.seq_len - 1:
            raise ValueError(f"Expected context length {self.seq_len - 1}, got {len(context)}")
        
        context_tuple = tuple(context)
        higher_order_counts = self.counts[context_tuple]
        context_count = self.context_counts[context_tuple]
        
        if context_count == 0:
            # Fallback to uniform distribution if context_count is zero
            return np.ones(self.vocab_size) / self.vocab_size
        
        higher_order_probs = np.maximum(higher_order_counts - self.discount, 0) / context_count
        lower_order_probs = self.continuation_counts / self.total_count
        lambda_weight = self.discount * np.count_nonzero(higher_order_counts) / context_count
        probs = higher_order_probs + lambda_weight * lower_order_probs
        
        return probs / np.sum(probs)  # Normalize to ensure it sums to 1

def load_data(filename: str) -> List[int]:
    """
    Load and tokenize data from a file.

    Args:
        filename (str): Path to the input file

    Returns:
        List[int]: List of tokenized data
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    with open(filename, 'r') as f:
        text = f.read()
    
    vocab = sorted(list(set(text)))
    char_to_token = {c: i for i, c in enumerate(vocab)}
    
    return [char_to_token[c] for c in text]

def evaluate_model(model: KneserNeyNgramModel, data: List[int]) -> Tuple[float, float]:
    """
    Evaluate the model on given data.

    Args:
        model (KneserNeyNgramModel): Trained n-gram model
        data (List[int]): Tokenized data for evaluation

    Returns:
        Tuple[float, float]: Loss and perplexity
    """
    sum_log_prob = 0.0
    count = 0
    for i in range(len(data) - model.seq_len + 1):
        context = data[i:i+model.seq_len-1]
        target = data[i+model.seq_len-1]
        probs = model.get_probs(context)
        sum_log_prob -= np.log(probs[target])
        count += 1
    
    avg_loss = sum_log_prob / count if count > 0 else 0.0
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

def main():
    data_dir = "data"
    train_file = os.path.join(data_dir, "train.txt")
    val_file = os.path.join(data_dir, "val.txt")
    test_file = os.path.join(data_dir, "test.txt")

    train_data = load_data(train_file)
    val_data = load_data(val_file)
    test_data = load_data(test_file)
    
    vocab_size = max(max(train_data), max(val_data), max(test_data)) + 1
    
    seq_lens = [3, 4, 5, 6]  # Sequence lengths to test
    discounts = [0.1, 0.3, 0.5, 0.7, 0.9]  # Discount values to test
    
    best_model = None
    best_val_loss = float('inf')
    
    for seq_len in seq_lens:
        for discount in discounts:
            model = KneserNeyNgramModel(vocab_size, seq_len, discount)
            
            # Train the model on the training data
            for i in range(len(train_data) - seq_len + 1):
                model.train(train_data[i:i+seq_len])
            
            # Evaluate the model on the validation data
            val_loss, val_perplexity = evaluate_model(model, val_data)
            print(f"seq_len: {seq_len}, discount: {discount:.2f}, val_loss: {val_loss:.4f}, val_perplexity: {val_perplexity:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
    
    if best_model is not None:
        # Evaluate the best model on the test data
        test_loss, test_perplexity = evaluate_model(best_model, test_data)
        print(f"\nBest model - seq_len: {best_model.seq_len}, discount: {best_model.discount:.2f}")
        print(f"Test loss: {test_loss:.4f}, Test perplexity: {test_perplexity:.4f}")
    else:
        print("No valid model found.")

if __name__ == "__main__":
    main()
