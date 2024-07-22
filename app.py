import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

probs = np.load("dev/ngram_probs.npy")

chars = ['\n'] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

def get_probs(context):
    indices = [char_to_idx[c] for c in context]
    return probs[tuple(indices)]

def sample_next_char(probs):
    return np.random.choice(chars, p=probs)

def generate_text(seed, length):
    generated_text = seed
    context = seed

    for _ in range(length):
        next_probs = get_probs(context)
        next_char = sample_next_char(next_probs)
        generated_text += next_char
        context = context[1:] + next_char

    return generated_text

def plot_probabilities(context):
    next_probs = get_probs(context)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=chars, y=next_probs)
    plt.title(f"Next Character Probabilities for Context: '{context}'")
    plt.xlabel("Characters")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    return plt

st.title(" 4-gram Language Model Visualizer")

st.sidebar.header("Model Information")
st.sidebar.write("Sequence Length: 4")
st.sidebar.write("Vocabulary Size:", len(chars))
st.sidebar.write("Using pre-trained probabilities from ngram_probs.npy")

st.header("Text Generation")
seed = st.text_input("Enter a seed text (3 characters):", value="the")
if len(seed) != 3:
    st.warning("Please enter exactly 3 characters for the seed.")
else:
    length = st.slider("Select generation length:", 10, 200, 50)
    if st.button("Generate Text"):
        generated_text = generate_text(seed, length)
        st.write("Generated Text:")
        st.write(generated_text)

st.header("Probability Visualization")
context = st.text_input("Enter a context (3 characters):", value="the")
if len(context) != 3:
    st.warning("Please enter exactly 3 characters for the context.")
else:
    fig = plot_probabilities(context)
    st.pyplot(fig)

st.header("How It Works")
st.write("""
1.  The 4-gram model predicts the next character using the previous 3 characters as context.
2.  It uses probabilities from 'ngram_probs.npy'.
3.  A bar chart displays the probability distribution for the next character.
4.  Characters are sampled based on these probabilities during text generation.
5.  The process repeats with the last 3 characters as the new context.
""")

st.header("Experiment")
st.write("""
Try various to see how the model behaves:
- Use common English trigrams like "the", "and", "ing" as seeds or contexts.
""")