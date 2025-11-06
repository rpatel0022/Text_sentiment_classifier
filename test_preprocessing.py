"""
Test script for data preprocessing
"""

import sys
from datasets import load_dataset
import re
from collections import Counter

print("=" * 60)
print("Testing IMDB Dataset Loading and Preprocessing")
print("=" * 60)

# Load IMDB dataset
print("\n1. Loading IMDB dataset...")
try:
    dataset = load_dataset("imdb")
    train_data = dataset["train"]
    test_data = dataset["test"]
    print("   ✓ Dataset loaded successfully!")
    print(f"   Train size: {len(train_data)}, Test size: {len(test_data)}")
except Exception as e:
    print(f"   ✗ Error loading dataset: {e}")
    sys.exit(1)

# Tokenizer
print("\n2. Initializing tokenizer...")


def tokenizer(text):
    """Basic English tokenizer"""
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


print("   ✓ Tokenizer initialized!")

# Test tokenizer with sample text
sample_text = "This movie is absolutely amazing!"
tokens = tokenizer(sample_text)
print(f"\n   Sample text: '{sample_text}'")
print(f"   Tokens: {tokens}")

# Build vocabulary
print("\n3. Building vocabulary from training data...")
print("   (This may take a minute...)")


def yield_tokens(data):
    """Generator to yield tokens from dataset"""
    for item in data:
        yield tokenizer(item["text"])


try:
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

    print(f"   ✓ Vocabulary built successfully!")
    print(f"   Vocabulary size: {len(vocab)}")
except Exception as e:
    print(f"   ✗ Error building vocabulary: {e}")
    sys.exit(1)

# Create pipelines
print("\n4. Creating text and label pipelines...")


def text_pipeline(text):
    """Convert text to list of token indices"""
    tokens = tokenizer(text)
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]


def label_pipeline(label):
    """Convert label to binary (0 or 1)"""
    return int(label)


print("   ✓ Pipelines created!")

# Test pipelines
print("\n5. Testing pipelines...")
test_text = "This is a great movie with excellent acting"
test_indices = text_pipeline(test_text)
print(f"\n   Test text: '{test_text}'")
print(f"   Token indices: {test_indices[:10]}... (showing first 10)")

test_label_pos = label_pipeline(1)
test_label_neg = label_pipeline(0)
print(f"\n   Label 1 (positive) -> {test_label_pos}")
print(f"   Label 0 (negative) -> {test_label_neg}")

# Test with actual dataset samples
print("\n6. Testing with actual dataset samples...")

print("\n   First 3 training samples:")
for i in range(3):
    item = train_data[i]
    text = item["text"]
    label = item["label"]

    processed_label = label_pipeline(label)
    processed_text = text_pipeline(text)

    label_name = "positive" if label == 1 else "negative"

    print(f"\n   Sample {i+1}:")
    print(f"   - Label: {label} ({label_name}) -> {processed_label}")
    print(f"   - Text preview: {text[:100]}...")
    print(f"   - Tokenized length: {len(processed_text)} tokens")

print("\n" + "=" * 60)
print("All tests passed successfully! ✓")
print("=" * 60)
