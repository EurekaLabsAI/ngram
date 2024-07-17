import ast
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# add the current work directory to access `ngram.py`
sys.path.append(".")

from ngram import (
    NgramModel,
    dataloader,
    eval_split,
    sample_discrete,
    RNG
)

# set the webpage title and favicon
st.set_page_config("N-gram Language Model", page_icon="🧮")

st.write("""# N-gram Language Model

Every combination of `Sequence Length` and `Smoothings` you set will
be evaluated to determine the best combination. The model will then
be retrained using the optimal hyperparameters and generate a heat
map showing conditional probabilities of the N-gram model.
""")

st.sidebar.write("## Parameters")
seq_lens_text = st.sidebar.text_input("Sequence Lengths", "[3, 4, 5]")
smoothings_text = st.sidebar.text_input("Smoothings", "[0.03, 0.1, 0.3, 1.0]")

# validate text inputs are Python lists, and show any error to the user
try:
    seq_lens = ast.literal_eval(seq_lens_text)
    smoothings = ast.literal_eval(smoothings_text)
except (SyntaxError, ValueError):
    st.sidebar.error(
        "sequence lengths and smoothings must be valid python syntax."
    )
    st.stop()

if not isinstance(seq_lens, list) or not isinstance(smoothings, list):
    st.sidebar.error("sequence lengths and smoothings must be python lists.")
    st.stop()

if not all(isinstance(x, int) and x > 0 for x in seq_lens):
    st.sidebar.error("sequence Lengths must be positive integers.")
    st.stop()

if not all(isinstance(x, (int, float)) and x > 0 for x in smoothings):
    st.sidebar.error("smoothing values must be positive numbers.")
    st.stop()

random_seed = st.sidebar.number_input("Random Seed", 1337)
test_tokens_path = st.sidebar.text_input("Test Tokens Path", "data/test.txt")
val_tokens_path = st.sidebar.text_input("Validation Tokens Path", "data/val.txt")
train_tokens_path = st.sidebar.text_input("Training Tokens Path", "data/train.txt")

iterations = len(seq_lens) * len(smoothings)
st.sidebar.write(f"Iterations: {iterations}")

if st.button("Evaluate Hyperparameters"):
    # "train" the Tokenizer, so we're able to map between characters and tokens
    train_text = open(train_tokens_path, "r").read()
    assert all(c == "\n" or ("a" <= c <= "z") for c in train_text)
    uchars = sorted(list(set(train_text)))  # unique characters we see in the input
    vocab_size = len(uchars)
    char_to_token = {c: i for i, c in enumerate(uchars)}
    token_to_char = {i: c for i, c in enumerate(uchars)}
    # designate \n as the delimiting <|endoftext|> token
    EOT_TOKEN = char_to_token["\n"]
    # pre-tokenize all the splits one time up here
    test_tokens = [char_to_token[c] for c in open(test_tokens_path, "r").read()]
    val_tokens = [char_to_token[c] for c in open(val_tokens_path, "r").read()]
    train_tokens = [char_to_token[c] for c in open(train_tokens_path, "r").read()]

    best_loss = float("inf")
    best_kwargs = {}

    st.write("## Hyperparameter Evaluation Results")
    progress_bar = st.progress(0.0, "")
    # set a placeholder to update the dataframe table in-place for each run
    df_placeholder = st.empty()
    df = pd.DataFrame(
        columns=[
            "Sequence Length",
            "Smoothing",
            "Training Loss",
            "Validation Loss",
        ]
    )

    for i, (seq_len, smoothing) in enumerate(itertools.product(seq_lens, smoothings)):
        # train the n-gram model
        model = NgramModel(vocab_size, seq_len, smoothing)
        for tape in dataloader(train_tokens, seq_len):
            model.train(tape)
        # evaluate the train/val loss
        train_loss = eval_split(model, train_tokens)
        val_loss = eval_split(model, val_tokens)
        # move the progress bar one step forward
        progress_bar.progress((i + 1) / (iterations), f"Iteration {i + 1} of {iterations}")
        # add a row onto the dataframe
        df.loc[len(df)] = {
            "Sequence Length": seq_len,
            "Smoothing": smoothing,
            "Training Loss": train_loss,
            "Validation Loss": val_loss,
        }
        # update the placeholder with the current dataframe
        df_placeholder.dataframe(df, height=int(35.2 * (iterations + 1)))
        # update the best hyperparameters
        if val_loss < best_loss:
            best_loss = val_loss
            best_kwargs = {"seq_len": seq_len, "smoothing": smoothing}

    st.write("## Best Hyperparameters")
    st.dataframe({
            "Hyperparameter": ["Sequence Length", "Smoothing"],
            "Value": [seq_len, smoothing],
        })

    # re-train the model with the best hyperparameters
    seq_len = best_kwargs["seq_len"]

    model = NgramModel(vocab_size, **best_kwargs)
    for tape in dataloader(train_tokens, seq_len):
        model.train(tape)

    # sample from the model
    sample_rng = RNG(random_seed)
    tape = [EOT_TOKEN] * (seq_len - 1)
    names = ""
    st.write("## Sample from the model")
    for _ in range(200):
        probs = model(tape)
        # sample the next token
        coinf = sample_rng.random()
        probs_list = probs.tolist()
        next_token = sample_discrete(probs_list, coinf)
        # otherwise update the token tape, print token and continue
        next_char = token_to_char[next_token]
        # update the tape
        tape.append(next_token)
        if len(tape) > seq_len - 1:
            tape = tape[1:]
        names += next_char
    st.text_area("", names, height=300)

    # at the end, evaluate and report the test loss
    st.write("## Model Evaluation Results")
    test_loss = eval_split(model, test_tokens)
    test_perplexity = np.exp(test_loss)
    df = pd.DataFrame(
        {
            "Metric": ["Test Loss", "Test Perplexity"],
            "Value": [test_loss, test_perplexity],
        }
    )
    st.dataframe(df)

    # get the final counts, and normalize them to probs
    counts = model.counts + model.smoothing
    probs = counts / counts.sum(axis=-1, keepdims=True)

    # TODO: allow for more shapes to mapped to a 2D space
    expected_shape = (27, 27, 27, 27)
    if probs.shape != (27, 27, 27, 27):
        st.error(
            f"unexpected probability matrix shape. expected {expected_shape}, but got {probs.shape}."
        )
        st.stop()

    # reshape to a 2D matrix and plot a heatmap
    reshaped = probs.reshape(27**2, 27**2)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(reshaped, cmap="hot", interpolation="nearest")
    ax.axis("off")

    # legend colorbar to show the probability of each color
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Probability", rotation=270, labelpad=15)

    st.pyplot(fig)

    st.write("""
    This heatmap visualizes the conditional probabilities of the N-gram model.

    Each pixel represents the probability of a specific sequence of characters occurring, 
    given the preceding characters. The color intensity indicates the likelihood:
    - Brighter (hotter) colors represent higher probabilities
    - Darker colors represent lower probabilities

    The x and y axes represent different character sequences. Due to the high dimensionality 
    of the data (27^4 possible sequences for a 4-gram model), the heatmap is a 2D projection 
    of this 4D space.

    Interpreting the heatmap:
    - Bright spots indicate common character sequences in the training data
    - Dark areas show rare or unlikely sequences
    - Patterns and clusters can reveal linguistic structures captured by the model

    This visualization helps to understand how the model has learned the statistical 
    patterns of the language from the training data.
    """)
