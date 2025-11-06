import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import time

# Import our modules
from data_preprocessing import (
    dataset,
    train_data,
    test_data,
    vocab,
    text_pipeline,
    label_pipeline,
)
from model import SentimentRNN


class IMDBDataset(Dataset):
    """Custom Dataset for IMDB reviews"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["label"], item["text"]


def collate_batch(batch):
    """Collate function to pad sequences in a batch"""
    labels, texts = [], []
    for label, text in batch:
        labels.append(label_pipeline(label))
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        texts.append(processed_text)

    # Pad sequences to the same length
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    labels = torch.tensor(labels, dtype=torch.float32)
    return texts, labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training")
    for texts, labels in progress_bar:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate predictions
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(labels.cpu().detach().numpy())

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for texts, labels in progress_bar:
            texts, labels = texts.to(device), labels.to(device)

            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate predictions
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels


def main():
    print("=" * 60)
    print("Training Sentiment Classifier")
    print("=" * 60)

    # Hyperparameters
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    OUTPUT_DIM = 1
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 3

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = IMDBDataset(train_data)
    test_dataset = IMDBDataset(test_data)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch
    )

    print(f"Train batches: {len(train_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")

    # Initialize model
    print(f"\nInitializing model...")
    print(f"Vocabulary size: {len(vocab)}")
    model = SentimentRNN(len(vocab), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    best_test_acc = 0.0
    training_history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_dataloader, criterion, optimizer, device
        )

        # Evaluate
        test_loss, test_acc, _, _ = evaluate(model, test_dataloader, criterion, device)

        # Save history
        training_history["train_loss"].append(train_loss)
        training_history["train_acc"].append(train_acc)
        training_history["test_loss"].append(test_loss)
        training_history["test_acc"].append(test_acc)

        # Print results
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc*100:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  ✓ New best model saved! (Acc: {best_test_acc*100:.2f}%)")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    # Load best model
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_dataloader, criterion, device
    )

    print(f"\nBest Test Accuracy: {test_acc*100:.2f}%")
    print(f"Best Test Loss: {test_loss:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            test_labels, test_preds, target_names=["Negative", "Positive"]
        )
    )

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": len(vocab),
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": OUTPUT_DIM,
            "training_history": training_history,
        },
        "sentiment_model.pt",
    )

    print("\n✓ Model saved as 'sentiment_model.pt'")
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
