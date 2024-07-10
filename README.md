# ngram

In this module we build the n-gram Language Model. In the process, we learn a lot of the basics of machine learning (training, evaluation, data splits, hyperparameters, overfitting) and the basics of autoregressive language modeling (tokenization, next token prediction, perplexity, sampling). GPT is "just" a very large n-gram model, too. The only difference is that GPT uses a neural network to calculate the probability of the next token, while n-gram uses a simple count-based approach.

In particular, running the final script will take the training set of names inside `train.txt`, train a model on it, and then sample new names. Here are example names that the model generates:

```
felton
jasiel
chaseth
nebjnvfobzadon
brittan
shir
...
```
