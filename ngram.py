"""
n-gram Language Model

Good reference:
Speech and Language Processing. Daniel Jurafsky & James H. Martin.
https://web.stanford.edu/~jurafsky/slp3/3.pdf

Example run:
python ngram.py
"""

import os
import itertools
import numpy as np

# -----------------------------------------------------------------------------
# random number generation

# class that mimics the random interface in Python, fully deterministic,
# and in a way that we also control fully, and can also use in C, etc.
class RNG:
    def __init__(self, seed):
        self.state = seed

    def random_u32(self):
        # xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        # doing & 0xFFFFFFFFFFFFFFFF is the same as cast to uint64 in C
        # doing & 0xFFFFFFFF is the same as cast to uint32 in C
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        # random float32 in [0, 1)
        return (self.random_u32() >> 8) / 16777216.0

random = RNG(1337)

# -----------------------------------------------------------------------------
# sampling from the model

def sample_discrete(probs, coinf):
    # sample from a discrete distribution
    cdf = 0.0
    for i, prob in enumerate(probs):
        cdf += prob
        if coinf < cdf:
            return i
    return len(probs) - 1  # in case of rounding errors

# -----------------------------------------------------------------------------
# models: n-gram model, and a fallback model that can use multiple n-gram models

class NgramModel:
    def __init__(self, vocab_size, seq_len, smoothing=0.0):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        # the parameters of this model: an n-dimensional array of counts
        self.counts = np.zeros((vocab_size,) * seq_len, dtype=np.uint32)
        # a buffer to store the uniform distribution, just to avoid creating it every time
        self.uniform = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size

    def train(self, tape):
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len
        self.counts[tuple(tape)] += 1

    def get_counts(self, tape):
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len - 1
        return self.counts[tuple(tape)]

    def __call__(self, tape):
        # returns the conditional probability distribution of the next token
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len - 1
        # get the counts, apply smoothing, and normalize to get the probabilities
        counts = self.counts[tuple(tape)].astype(np.float32)
        counts += self.smoothing # add smoothing ("fake counts") to all counts
        counts_sum = counts.sum()
        probs = counts / counts_sum if counts_sum > 0 else self.uniform
        return probs

# currently unused, just for illustration
class BackoffNgramModel:
    """
    A backoff model that can be used to combine multiple n-gram models of different orders.
    During training, it updates all the models with the same data.
    During inference, it uses the highest order model that has data for the current context.
    """
    def __init__(self, vocab_size, seq_len, smoothing=0.0, counts_threshold=0):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.counts_threshold = counts_threshold
        self.models = {i: NgramModel(vocab_size, i, smoothing) for i in range(1, seq_len + 1)}

    def train(self, tape):
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len
        for i in range(1, self.seq_len + 1):
            self.models[i].train(tape[-i:])

    def __call__(self, tape):
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len - 1
        # find the highest order model that has data for the current context
        for i in reversed(range(1, self.seq_len + 1)):
            tape_i = tape[-i+1:] if i > 1 else []
            counts = self.models[i].get_counts(tape_i)
            if counts.sum() > self.counts_threshold:
                return self.models[i](tape_i)
        # we shouldn't get here because unigram model should always have data
        raise ValueError("no model found for the current context")

# -----------------------------------------------------------------------------
# data iteration and evaluation utils

# small utility function to iterate tokens with a fixed-sized window
def dataloader(tokens, window_size):
    for i in range(len(tokens) - window_size + 1):
        yield tokens[i:i+window_size]

def eval_split(model, tokens):
    # evaluate a given model on a given sequence of tokens (splits, usually)
    sum_loss = 0.0
    count = 0
    for tape in dataloader(tokens, model.seq_len):
        x = tape[:-1] # the context
        y = tape[-1]  # the target
        probs = model(x)
        prob = probs[y]
        sum_loss += -np.log(prob)
        count += 1
    mean_loss = sum_loss / count if count > 0 else 0.0
    return mean_loss

# -----------------------------------------------------------------------------

# "train" the Tokenizer, so we're able to map between characters and tokens
train_text = open('data/train.txt', 'r').read()
assert all(c == '\n' or ('a' <= c <= 'z') for c in train_text)
uchars = sorted(list(set(train_text))) # unique characters we see in the input
vocab_size = len(uchars)
char_to_token = {c: i for i, c in enumerate(uchars)}
token_to_char = {i: c for i, c in enumerate(uchars)}
EOT_TOKEN = char_to_token['\n'] # designate \n as the delimiting <|endoftext|> token
# pre-tokenize all the splits one time up here
test_tokens = [char_to_token[c] for c in open('data/test.txt', 'r').read()]
val_tokens = [char_to_token[c] for c in open('data/val.txt', 'r').read()]
train_tokens = [char_to_token[c] for c in open('data/train.txt', 'r').read()]

# hyperparameter search with grid search over the validation set
seq_lens = [3, 4, 5]
smoothings = [0.03, 0.1, 0.3, 1.0]
best_loss = float('inf')
best_kwargs = {}
for seq_len, smoothing in itertools.product(seq_lens, smoothings):
    # train the n-gram model
    model = NgramModel(vocab_size, seq_len, smoothing)
    for tape in dataloader(train_tokens, seq_len):
        model.train(tape)
    # evaluate the train/val loss
    train_loss = eval_split(model, train_tokens)
    val_loss = eval_split(model, val_tokens)
    print("seq_len %d | smoothing %.2f | train_loss %.4f | val_loss %.4f"
          % (seq_len, smoothing, train_loss, val_loss))
    # update the best hyperparameters
    if val_loss < best_loss:
        best_loss = val_loss
        best_kwargs = {'seq_len': seq_len, 'smoothing': smoothing}

# re-train the model with the best hyperparameters
seq_len = best_kwargs['seq_len']
print("best hyperparameters:", best_kwargs)
model = NgramModel(vocab_size, **best_kwargs)
for tape in dataloader(train_tokens, seq_len):
    model.train(tape)

# sample from the model
tape = [EOT_TOKEN] * (seq_len - 1)
for _ in range(200):
    probs = model(tape)
    # sample the next token
    coinf = random.random()
    probs_list = probs.tolist()
    next_token = sample_discrete(probs_list, coinf)
    # otherwise update the token tape, print token and continue
    next_char = token_to_char[next_token]
    # update the tape
    tape.append(next_token)
    if len(tape) > seq_len - 1:
        tape = tape[1:]
    print(next_char, end='')
print() # newline

# at the end, evaluate and report the test loss
test_loss = eval_split(model, test_tokens)
test_perplexity = np.exp(test_loss)
print("test_loss %f, test_perplexity %f" % (test_loss, test_perplexity))

# get the final counts, normalize them to probs, and write to disk for vis
counts = model.counts + model.smoothing
probs = counts / counts.sum(axis=-1, keepdims=True)
vis_path = os.path.join("dev", "ngram_probs.npy")
np.save(vis_path, probs)
print(f"wrote {vis_path} to disk (for visualization)")
