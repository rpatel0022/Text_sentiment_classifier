import torch
from torch import nn


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        return self.sigmoid(self.fc(hidden[-1]))


# Model configuration
def get_model_config():
    """Returns default model configuration"""
    config = {
        "embed_dim": 100,  # Embedding dimension
        "hidden_dim": 256,  # LSTM hidden dimension
        "output_dim": 1,  # Binary classification (1 output)
    }
    return config


# Initialize model
def create_model(vocab_size, embed_dim=100, hidden_dim=256, output_dim=1):
    """Create and return a SentimentRNN model"""
    model = SentimentRNN(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    return model


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("Testing SentimentRNN Model")
    print("=" * 60)

    # Sample parameters
    vocab_size = 10000
    embed_dim = 100
    hidden_dim = 256
    output_dim = 1

    # Create model
    model = create_model(vocab_size, embed_dim, hidden_dim, output_dim)
    print(f"\n✓ Model created successfully!")
    print(f"\nModel Architecture:")
    print(model)

    # Test with dummy data
    batch_size = 32
    seq_length = 50
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))

    print(f"\n✓ Testing forward pass...")
    print(f"   Input shape: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n✓ Model Parameters:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    print("\n" + "=" * 60)
    print("Model test completed successfully! ✓")
    print("=" * 60)
