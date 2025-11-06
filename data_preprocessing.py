from datasets import load_dataset
import re
from collections import Counter

# Load IMDB dataset
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]
print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")


# Simple tokenizer (basic English)
def tokenizer(text):
    """Basic English tokenizer"""
    text = text.lower()
    # Split on spaces and punctuation
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


def yield_tokens(data):
    """Generator to yield tokens from dataset"""
    for item in data:
        yield tokenizer(item["text"])


# Build vocabulary
print("Building vocabulary...")
counter = Counter()
for tokens in yield_tokens(train_data):
    counter.update(tokens)

# Create vocabulary - keep words that appear at least 5 times
min_freq = 5
vocab = {"<unk>": 0, "<pad>": 1}
idx = 2
for word, count in counter.items():
    if count >= min_freq:
        vocab[word] = idx
        idx += 1

print(f"Vocabulary size: {len(vocab)}")

# Reverse vocabulary for lookup
idx_to_word = {idx: word for word, idx in vocab.items()}


# Text and label pipelines
def text_pipeline(text):
    """Convert text to list of token indices"""
    tokens = tokenizer(text)
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]


def label_pipeline(label):
    """Convert label to binary (0 or 1)"""
    return int(label)  # IMDB dataset already has 0/1 labels


# Example usage
if __name__ == "__main__":
    # Test with a sample
    sample_text = "This movie is absolutely amazing!"
    sample_indices = text_pipeline(sample_text)
    print(f"\nSample text: {sample_text}")
    print(f"Token indices: {sample_indices}")
