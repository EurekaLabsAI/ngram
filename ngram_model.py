import numpy as np

class NgramModel:
    def __init__(self, vocab_size, seq_len, smoothing=0.1):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.counts = {}
        self.uniform = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size

    def train(self, tape):
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len
        key = tuple(tape)
        if key in self.counts:
            self.counts[key] += 1
        else:
            self.counts[key] = 1

    def get_counts(self, tape):
        return self.counts.get(tuple(tape), 0)

    def __call__(self, tape):
        assert isinstance(tape, list)
        assert len(tape) == self.seq_len - 1
        context_key = tuple(tape)
        counts = np.zeros(self.vocab_size, dtype=np.float32)
        
        for i in range(self.vocab_size):
            counts[i] = self.get_counts(list(context_key) + [i])
        
        counts += self.smoothing
        counts_sum = counts.sum()
        probs = counts / counts_sum if counts_sum > 0 else self.uniform
        
        return probs if counts_sum != 0 else self.uniform

    def predict_next(self, context):
        probs = self(context)
        return np.argmax(probs)
