import torch
from tqdm import tqdm
import torch.optim as optim
import time
from .features import FeatureSpace
from .metric import Snack
from .data import load_msf_data


def get_device():
    """Determine the best available device for PyTorch"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    else:
        return torch.device("cpu")


def train_model(
    model, dataset_path, num_epochs=10, learning_rate=0.001, batch_size=4, device=None
):
    """
    Train the SNACK model with acceleration using GPU/MPS if available.

    Args:
        model: The SNACK model to train
        dataset_path: Path to the dataset
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        batch_size: Number of sequence pairs to process in a batch
        device: Device to use for training (if None, will be auto-detected)
    """
    # Auto-detect device if not provided
    if device is None:
        device = get_device()

    print(f"Training on device: {device}")

    # Move model to device
    model = model.to(device)

    # Load data
    alignment_data = load_msf_data(dataset_path)

    # Sort by sequence length for more efficient batch processing
    # This groups similar length sequences together to minimize padding
    alignment_data.sort(key=lambda pair: (len(pair[0]), len(pair[1])))

    # Define optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Setup learning rate scheduler to reduce LR if training plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # Training loop
    model.train()
    total_start_time = time.time()
    best_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        batch_count = 0

        # Process data in batches
        for i in tqdm(range(0, len(alignment_data), batch_size)):
            batch = alignment_data[i : i + batch_size]
            batch_count += 1

            # Zero gradients
            optimizer.zero_grad()

            # Compute loss for the batch
            loss = model.total_loss(batch)

            # Skip problematic batches if they cause errors
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Skipping batch {batch_count} with invalid loss value")
                continue

            # Backpropagation
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Parameter update
            optimizer.step()

            # Track loss
            total_loss += loss.item()

            # Print progress for larger batches
            if batch_count % 10 == 0:
                avg_loss = total_loss / batch_count
                print(
                    f"  Batch {batch_count}, Progress: {i+len(batch)}/{len(alignment_data)} pairs, Loss: {avg_loss:.4f}"
                )

        # Average loss for the epoch
        avg_epoch_loss = total_loss / (len(alignment_data) / batch_size)

        # Update learning rate based on loss
        scheduler.step(avg_epoch_loss)

        # Track training time
        epoch_time = time.time() - epoch_start_time
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Loss: {avg_epoch_loss:.4f}, "
            f"Time: {epoch_time:.2f}s, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Save best model (could add checkpoint saving here)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"New best loss: {best_loss:.4f}")
            # Optionally save model: torch.save(model.state_dict(), "best_model.pt")

    total_time = time.time() - total_start_time
    print(f"Training completed in {total_time:.2f}s")


if __name__ == "__main__":
    import os

    # Get the current directory and use relative path for dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(os.path.dirname(current_dir), "data")

    # Auto-detect the best available device
    device = get_device()
    print(f"Using device: {device}")

    # Инициализируем пространство признаков
    feature_space = FeatureSpace()

    # Инициализируем модель с метрикой
    metric = Snack(feature_space=feature_space)

    # Запускаем обучение с использованием GPU/MPS если доступно
    train_model(
        model=metric,
        dataset_path=dataset_path,
        num_epochs=10,
        learning_rate=0.001,
        batch_size=4,
        device=device,
    )
