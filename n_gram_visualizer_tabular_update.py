import streamlit as st
import numpy as np
import itertools
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from Levenshtein import distance as levenshtein_distance
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
import pandas as pd

# ----------------------------------------------------------------------
# Custom CSS for Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
        padding: 20px;
    }
    h1, h2, h3, h4, h5 {
        color: #343a40;
        font-weight: 600;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton>button, .stSlider, .stTextInput {
        border-radius: 10px;
        padding: 8px;
        background-color: #e0e0e0;
        color: #212529;
        border: none;
        transition: background-color 0.3s, box-shadow 0.3s;
    }
    .stButton>button:hover, .stSlider:hover {
        background-color: #bdbdbd;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .dataframe, .stMarkdown, .stText, .stPlotlyChart {
        border-radius: 10px;
        padding: 10px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .title-container {
        text-align: center;
        margin-bottom: 20px;
        white-space: nowrap;
    }
    .title-container h1 {
        font-size: 2em;
        color: #1c1e21;
        display: inline-block;
        margin: 0;
        white-space: nowrap;
    }
    .instructions {
        background-color: #f1f3f5;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .instructions h3 {
        color: #343a40;
    }
    .instructions p {
        color: #495057;
        font-size: 1.1em;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Streamlit App Title
st.markdown("""
<div class="title-container">
    <h1>N-Gram Language Model Visualization</h1>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Instructions for Using the App
st.markdown("""
<div class="instructions">
    <h3>🔧 How to Use the N-Gram Language Model App</h3>
    <p>Welcome to the N-Gram Language Model Visualization app! Here's how you can use it:</p>
    <ol>
        <li><b>Upload your training, validation, and test files</b> using the sidebar.</li>
        <li>Adjust <b>hyperparameters</b> for the model, including sequence length and smoothing.</li>
        <li>Watch the model training process, see the losses, and identify the best hyperparameters.</li>
        <li>Use the app to visualize the <b>confusion matrix, top N predicted tokens</b>, and more!</li>
        <li>Explore additional features like <b>animated text generation</b> and a <b>word cloud</b> of generated tokens.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Sidebar Parameters
st.sidebar.header("Model Parameters")
seq_len = st.sidebar.slider("Initial Sequence Length (N-gram)", 2, 10, 4)
smoothing = st.sidebar.slider("Initial Smoothing Value", 0.0, 1.0, 0.1, 0.01)
animation_speed = st.sidebar.slider("Animation Speed (seconds per step)", 0.01, 0.5, 0.1)

# File Upload for Training, Validation, and Testing Sets
train_file = st.sidebar.file_uploader("📂 Upload Training File", type=["txt"])
val_file = st.sidebar.file_uploader("📂 Upload Validation File", type=["txt"])
test_file = st.sidebar.file_uploader("📂 Upload Test File", type=["txt"])

# Ensure all files are uploaded
if train_file is None or val_file is None or test_file is None:
    st.warning("⚠️ Please upload all three files: training, validation, and test files.")
    st.stop()


# ----------------------------------------------------------------------
# Load Training Data
@st.cache_data
def load_data(train_file, val_file, test_file):
    train_text = train_file.read().decode("utf-8")
    val_text = val_file.read().decode("utf-8")
    test_text = test_file.read().decode("utf-8")
    return train_text, val_text, test_text

train_text, val_text, test_text = load_data(train_file, val_file, test_file)

# Unique character processing for vocab
unique_chars = sorted(set(train_text + val_text + test_text))
vocab_size = len(unique_chars)
char_to_token = {char: idx for idx, char in enumerate(unique_chars)}
token_to_char = {idx: char for idx, char in enumerate(unique_chars)}

# Tokenize data
def tokenize(text):
    return [char_to_token[char] for char in text if char in char_to_token]

train_tokens = tokenize(train_text)
val_tokens = tokenize(val_text)
test_tokens = tokenize(test_text)

# ----------------------------------------------------------------------
# Helper Classes and Functions
class RNG:
    def __init__(self, seed):
        self.state = seed

    def random_u32(self):
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        return (self.random_u32() >> 8) / 16777216.0

def sample_discrete(probs, coinf):
    cdf = 0.0
    for i, prob in enumerate(probs):
        cdf += prob
        if coinf < cdf:
            return i
    return len(probs) - 1

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

# Levenshtein Integration for Comparison
def compare_generated_text(reference, generated):
    lev_distance = levenshtein_distance(reference, generated)
    st.write(f"Levenshtein Distance: {lev_distance}")
    st.write(f"Generated Text: {generated}")

# Dataloader & Evaluation
def dataloader(tokens, window_size):
    for i in range(len(tokens) - window_size + 1):
        yield tokens[i:i+window_size]

def eval_split(model, tokens):
    sum_loss = 0.0
    count = 0
    for tape in dataloader(tokens, model.seq_len):
        x = tape[:-1]
        y = tape[-1]
        probs = model(x)
        prob = probs[y]
        sum_loss += -np.log(prob)
        count += 1
    return sum_loss / count if count > 0 else 0.0

# ----------------------------------------------------------------------
# Visualization & More (Keep the rest of your original visualizations)


def plot_confusion_matrix(true_labels, predicted_labels, labels):
    char_labels = [token_to_char[idx] for idx in labels]
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=char_labels, yticklabels=char_labels)
    plt.xlabel("Predicted Characters")
    plt.ylabel("Actual Characters")
    plt.title("Confusion Matrix of Token Predictions")
    st.pyplot(plt)

def plot_top_n_predictions(model, context, N=5):
    probs = model(context)
    top_indices = np.argsort(probs)[-N:][::-1]
    top_probs = probs[top_indices]
    tokens = [token_to_char[idx] for idx in top_indices]
    plt.figure(figsize=(8, 6))
    plt.bar(tokens, top_probs, color='teal')
    plt.title("Top N Predicted Tokens")
    plt.xlabel("Token")
    plt.ylabel("Probability")
    st.pyplot(plt)

def generate_word_cloud(text):
    if not text.strip():
        st.warning("No text generated for the word cloud. Please try generating more content.")
        return
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Hyperparameter Tuning and Training (UPDATED)
st.subheader("🎨 Hyperparameter Tuning Results")
seq_lens = [3, 4, 5]
smoothings = [0.03, 0.1, 0.3, 1.0]
best_loss = float('inf')
best_params = {'seq_len': None, 'smoothing': None}
results = []

# Hyperparameter combinations
# Hyperparameter Tuning (REAL-TIME UPDATES)
@st.cache_data(show_spinner=False)
def hyperparameter_tuning_real_time(seq_lens, smoothings):
    best_loss, best_train_loss, best_params = float('inf'), float('inf'), {'seq_len': None, 'smoothing': None}
    results = []
    results_table = st.empty()  # Placeholder to update the table dynamically
    progress_bar = st.progress(0)  # Progress bar to show completion progress
    total_combinations = len(seq_lens) * len(smoothings)
    completed = 0

    for seq_len_trial, smoothing_trial in itertools.product(seq_lens, smoothings):
        trial_model = NgramModel(vocab_size, seq_len_trial, smoothing_trial)
        for tape in dataloader(train_tokens, seq_len_trial):
            trial_model.train(tape)
        train_loss = eval_split(trial_model, train_tokens)
        val_loss = eval_split(trial_model, val_tokens)
        
        # Save results to display in a table
        results.append({
            'Sequence Length': seq_len_trial,
            'Smoothing': smoothing_trial,
            'Training Loss': round(train_loss, 4),
            'Validation Loss': round(val_loss, 4)
        })
        
        # Update best parameters based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_train_loss = train_loss
            best_params = {'seq_len': seq_len_trial, 'smoothing': smoothing_trial}

        # Update progress and results table
        completed += 1
        progress_bar.progress(completed / total_combinations)
        results_df = pd.DataFrame(results)
        results_table.table(results_df)  # Dynamically update the table

    return best_params, best_train_loss, best_loss

# Run hyperparameter tuning and retrieve results
# Run hyperparameter tuning and retrieve results
best_params, best_train_loss, best_loss = hyperparameter_tuning_real_time(seq_lens, smoothings)

# Display the best hyperparameters and corresponding losses
st.write(f"⭐ Best Hyperparameters: Sequence Length = {best_params['seq_len']}, Smoothing = {best_params['smoothing']}")
st.write(f"Training Loss with Best Parameters: {best_train_loss:.4f}")
st.write(f"Validation Loss with Best Parameters: {best_loss:.4f}")

# Evaluate on Test Set
st.subheader("🧪 Evaluating on Test Set with Best Hyperparameters")
best_model = NgramModel(vocab_size, best_params['seq_len'], best_params['smoothing'])
for tape in dataloader(train_tokens, best_params['seq_len']):
    best_model.train(tape)

# Test evaluation
test_loss = eval_split(best_model, test_tokens)
test_perplexity = np.exp(test_loss)
st.write(f"Test Loss: {test_loss:.4f}")
st.write(f"Test Perplexity: {test_perplexity:.4f}")

# Save Model
model_path = "ngram_model_best.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
st.success(f"Model saved as '{model_path}'")

# Animated Probability Plot for Text Generation on Test Data
st.subheader("Generated Text with Probability Animation")
sample_rng = RNG(1337)
tape = [char_to_token[test_text[0]]] * (best_params['seq_len'] - 1)
generated_text = ""

placeholder_text = st.empty()
placeholder_bar_chart = st.empty()
gen_loss_fig, gen_loss_ax = plt.subplots()
gen_loss_plot = st.pyplot(gen_loss_fig)
gen_losses = []

for i in range(100):
    probs = best_model(tape)
    next_token = sample_discrete(probs.tolist(), sample_rng.random())
    generated_name = token_to_char.get(next_token, "")
    generated_text += generated_name

    gen_loss = -np.log(probs[next_token])
    gen_losses.append((i + 1, gen_loss))

    tape.append(next_token)
    if len(tape) > best_params['seq_len'] - 1:
        tape.pop(0)

    placeholder_text.text(generated_text.strip())

    fig, ax = plt.subplots()
    ax.bar(range(vocab_size), probs, color='skyblue')
    ax.set_title("Token Probability Distribution")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Token ID")
    ax.set_ylabel("Probability")
    ax.set_xticks(range(vocab_size))
    ax.set_xticklabels([token_to_char.get(i, '') for i in range(vocab_size)], rotation=90)

    placeholder_bar_chart.pyplot(fig)
    plt.close(fig)

    gen_loss_ax.clear()
    gen_loss_ax.plot([x[0] for x in gen_losses], [x[1] for x in gen_losses], 'ro-', label='Generation Loss')
    gen_loss_ax.set_ylim(0, 10)
    gen_loss_ax.set_title("Generation Loss Over Steps")
    gen_loss_ax.set_xlabel("Generation Step")
    gen_loss_ax.set_ylabel("Loss")
    gen_loss_ax.legend()
    gen_loss_plot.pyplot(gen_loss_fig)
    plt.close(gen_loss_fig)
    time.sleep(animation_speed)

# New Visualizations
st.subheader("🧩 Confusion Matrix for Token Predictions")
true_tokens, predicted_tokens = [], []
for tape in dataloader(test_tokens, best_params['seq_len']):
    true_token = tape[-1]
    predicted_probs = best_model(tape[:-1])
    predicted_token = np.argmax(predicted_probs)
    true_tokens.append(true_token)
    predicted_tokens.append(predicted_token)

plot_confusion_matrix(true_tokens, predicted_tokens, labels=list(range(vocab_size)))

st.subheader("🔎 Top N Predicted Tokens")
example_context = test_tokens[:best_params['seq_len']-1]
plot_top_n_predictions(best_model, example_context)

@st.cache_data
def generate_text_for_wordcloud(_model, num_chars=4000):
    generated_text = ""
    tape = [char_to_token[test_text[0]]] * (best_params['seq_len'] - 1)

    for _ in range(num_chars):
        probs = _model(tape)
        next_token = sample_discrete(probs.tolist(), RNG(1337).random())
        generated_text += token_to_char.get(next_token, "")
        tape.append(next_token)
        if len(tape) > best_params['seq_len'] - 1:
            tape.pop(0)

    return generated_text

st.subheader("🌥️ Word Cloud of Generated Tokens")
generated_text = generate_text_for_wordcloud(best_model, num_chars=5000)
generate_word_cloud(generated_text)

# Sidebar: Optional Reference Text for Comparison
reference_file = st.sidebar.file_uploader("📂 Upload Reference Text for Comparison (Optional)", type=["txt"], key="reference_file")

# Load Reference Text
reference_text_lines = []
if reference_file:
    raw_reference_text = reference_file.read().decode("utf-8").strip()
    if raw_reference_text:
        reference_text_lines = raw_reference_text.split("\n")
        st.write("Preview of First 5 Reference Names:", reference_text_lines[:5])
    else:
        st.warning("The reference file appears to be empty. Please upload a valid text file.")

# Levenshtein Distance Comparison Function for Multiple Comparisons
def compare_multiple_generated_with_references(generated_samples, reference_samples):
    if not reference_samples:
        st.warning("No reference texts provided for comparison.")
        return

    # Prepare data for the comparison table
    comparison_data = {
        "Generated Text": [],
        "Reference Text": [],
        "Levenshtein Distance": []
    }

    # Compare each generated text with each reference text and record the distance
    for gen in generated_samples:
        for ref in reference_samples:
            lev_distance = levenshtein_distance(ref, gen)
            comparison_data["Generated Text"].append(gen)
            comparison_data["Reference Text"].append(ref)
            comparison_data["Levenshtein Distance"].append(lev_distance)

    # Display the comparison as a table
    st.markdown(f"**Levenshtein Distance Comparison Table**")
    st.table(comparison_data)

# Generate Text and Compare
st.subheader("📝 Levenshtein Comparison with Reference Texts")
# Create multiple generated samples for demonstration
generated_samples = [
    "".join([token_to_char.get(idx, "") for idx in test_tokens[0:50]]),  # Sample 1
    "".join([token_to_char.get(idx, "") for idx in test_tokens[50:100]]),  # Sample 2
    "".join([token_to_char.get(idx, "") for idx in test_tokens[100:150]]),  # Sample 3
]

# Use the first three lines of the reference file for comparison, or mock data if not available
reference_samples = reference_text_lines[:3] if len(reference_text_lines) >= 3 else []

# Compare multiple generated samples with the reference samples
if generated_samples and reference_samples:
    compare_multiple_generated_with_references(generated_samples, reference_samples)
else:
    st.write("Generated Sample (Example):", generated_samples[0])
    st.info("Upload a reference text file from the sidebar to compare generated samples.")

# Visualization Function for Levenshtein Heatmap
def plot_levenshtein_heatmap(generated_texts, reference_texts):
    if not generated_texts or not reference_texts:
        st.warning("Cannot create heatmap without generated and reference texts.")
        return
    
    # Calculate distances between each generated text and each reference text
    distances = []
    for gen in generated_texts:
        row = [levenshtein_distance(gen, ref) for ref in reference_texts]
        distances.append(row)
    
    # Create and display the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(distances, annot=True, fmt="d", cmap="coolwarm",
                xticklabels=[f"Ref {i+1}" for i in range(len(reference_texts))],
                yticklabels=[f"Gen {i+1}" for i in range(len(generated_texts))])
    plt.title("Levenshtein Distance Heatmap")
    plt.xlabel("Reference Texts")
    plt.ylabel("Generated Texts")
    st.pyplot(plt)

# Display Heatmap if both generated and reference samples are available
if generated_samples and reference_samples:
    plot_levenshtein_heatmap(generated_samples, reference_samples)



import pandas as pd

def print_token_mappings():
    # Display total number of tokens
    total_tokens = len(char_to_token)
    st.write(f"Total Number of Tokens: {total_tokens}")

    # Prepare data for the table
    token_mappings = []
    for char, token in char_to_token.items():
        char_display = "_" if char == " " else char  # Show space as '_'
        token_mappings.append({"Character": char_display, "Token": token})

    # Convert to DataFrame and display as a table
    token_mappings_df = pd.DataFrame(token_mappings)
    st.table(token_mappings_df)

# Call this function before generating the confusion matrix
print_token_mappings()


