# ngram

In this module we build the n-gram Language Model. In the process, we learn a lot of the basics of machine learning (training, evaluation, data splits, hyperparameters, overfitting) and the basics of autoregressive language modeling (tokenization, next token prediction, perplexity, sampling). GPT is "just" a very large n-gram model, too. The only difference is that GPT uses a neural network to calculate the probability of the next token, while n-gram uses a simple count-based approach.

Our dataset is that of 32,032 names from [ssa.gov](https://www.ssa.gov/oact/babynames/) for the year 2018, which were split into 1,000 names in the test split, 1,000 in val split, and the rest in the training split, all of them inside the `data/` folder. Therefore, our n-gram model will essentially try to learn the statistics of the characters in these names, and then generate new names by sampling from the model.

A great reference for this module is [Chapter 3](https://web.stanford.edu/~jurafsky/slp3/3.pdf) of "Speech and Language Processing" by Jurafsky and Martin.

Currently, the best "build this repo from scratch" reference is the ["The spelled-out intro to language modeling: building makemore"](https://www.youtube.com/watch?v=PaCmpygFfXo) YouTube video, though some of the details have changed around a bit. The major departure is that the video covers a bigram Language Model, which for us is just a special case when `n = 2` for the n-gram.

### Python version

To run the Python code, ensure you have `numpy` installed (e.g. `pip install numpy`), and then run the script:

```bash
python ngram.py
```

You'll see that the script first "trains" a small character-level Tokenizer (the vocab size is 27 for all 26 lowercase English letters and the newline character), then it conducts a small grid search of n-gram models with various hyperparameter settings for the n-gram order `n` and the smoothing factor, using the validation split. With default settings on our data, the values that turn out to be optimal are `n=4, smoothing=0.1`. It then takes this best model, samples 200 characters from it, and finally reports the test loss and perplexity. Here is the full output, it should only take a few seconds to produce:

```
python ngram.py
seq_len 3 | smoothing 0.03 | train_loss 2.1843 | val_loss 2.2443
seq_len 3 | smoothing 0.10 | train_loss 2.1870 | val_loss 2.2401
seq_len 3 | smoothing 0.30 | train_loss 2.1935 | val_loss 2.2404
seq_len 3 | smoothing 1.00 | train_loss 2.2117 | val_loss 2.2521
seq_len 4 | smoothing 0.03 | train_loss 1.8703 | val_loss 2.1376
seq_len 4 | smoothing 0.10 | train_loss 1.9028 | val_loss 2.1118
seq_len 4 | smoothing 0.30 | train_loss 1.9677 | val_loss 2.1269
seq_len 4 | smoothing 1.00 | train_loss 2.1006 | val_loss 2.2114
seq_len 5 | smoothing 0.03 | train_loss 1.4955 | val_loss 2.3540
seq_len 5 | smoothing 0.10 | train_loss 1.6335 | val_loss 2.2814
seq_len 5 | smoothing 0.30 | train_loss 1.8610 | val_loss 2.3210
seq_len 5 | smoothing 1.00 | train_loss 2.2132 | val_loss 2.4903
best hyperparameters: {'seq_len': 4, 'smoothing': 0.1}
felton
jasiel
chaseth
nebjnvfobzadon
brittan
shir
esczsvn
freyanty
aubren
... (truncating) ...
test_loss 2.106370, test_perplexity 8.218358
wrote dev/ngram_probs.npy to disk (for visualization)
```

As you can see, the 4-gram model sampled some relatively reasonable names like "felton" and "jasiel", but also some weirder ones like "nebjnvfobzadon", but you can't expect too much from a little 4-gram character-level language model. Finally, the test perplexity is reported at ~8.2, so the model is as confused about every character in the test set as if it was choosing randomly from 8.2 equally likely characters.

The Python code also writes out the n-gram probabilities to disk into the `dev/` folder, which you can then inspect with the attached Jupyter notebook [dev/visualize_probs.ipynb](dev/visualize_probs.ipynb).

### C version

The C model is identical in functionality but skips the cross-validation. Instead, it hardcodes `n=4, smoothing=0.01`, but does the training, sampling, and test perplexity evaluation and achieves the exact same results as the Python version. An example of compiling and running the C code is as follows:

```bash
clang -O3 -o ngram ngram.c -lm
./ngram
```

The C version runs, of course, much faster. You'll see the same samples and test perplexity.

### TODOs

- Make better
- Make exercises
- Call for help: nice visualization / webapp that shows and animates the 4-gram language model and how it works.
