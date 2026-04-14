/*

  Varun Raghavendra

  Spring 2026

  CS 5330 Computer Vision

  Vision Transformer training and evaluation on MNIST

*/

import sys
import os
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from net_transformer import NetConfig, NetTransformer


OUTPUTS_DIR = "outputs/task4"
MODELS_DIR = "saved_models"
DATA_DIR = "./data"

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# Select the best available PyTorch device.
# Uses CUDA when available and falls back to CPU safely.
def get_device():
    if torch.cuda.is_available():
        try:
            _ = torch.cuda.current_device()
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        except Exception as exc:
            print(f"CUDA detected but unusable. Falling back to CPU. Reason: {exc}")
            return torch.device("cpu")

    print("CUDA not available. Using CPU.")
    return torch.device("cpu")


# Choose a suitable number of DataLoader workers.
# Scales worker count based on available CPU cores.
def get_num_workers():
    cpu_count = os.cpu_count() or 2
    return min(4, max(1, cpu_count // 2))


# Build training and testing MNIST dataloaders.
# Applies normalization and optional CUDA pin memory.
def get_data_loaders(batch_size=256, num_workers=2, device=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    use_pin_memory = (device is not None and device.type == "cuda")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    return train_loader, test_loader


# Evaluate the model on a dataset loader.
# Returns average loss and classification accuracy.
def evaluate_network(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=(device.type == "cuda"))
            target = target.to(device, non_blocking=(device.type == "cuda"))

            output = model(data)
            loss = F.nll_loss(output, target, reduction="sum")

            total_loss += loss.item()
            predictions = output.argmax(dim=1)
            total_correct += predictions.eq(target).sum().item()
            total_samples += target.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


# Train the transformer model for one epoch.
# Supports mixed precision training when CUDA is enabled.
def train_one_epoch(model, optimizer, loader, device, scaler, epoch, log_interval=100):
    model.train()
    running_loss = 0.0
    total_samples = 0

    use_amp = (device.type == "cuda")

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device, non_blocking=use_amp)
        target = target.to(device, non_blocking=use_amp)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = F.nll_loss(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        batch_size = target.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        if batch_idx % log_interval == 0:
            seen = batch_idx * len(data)
            total = len(loader.dataset)
            print(f"  Epoch {epoch} [{seen}/{total}] loss: {loss.item():.4f}")

    return running_loss / total_samples


# Plot training loss, test loss, and test accuracy curves.
# Saves the visualization to the outputs directory.
def plot_results(train_losses, test_losses, test_accs, tag):
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, marker="o", label="Train loss")
    ax1.plot(epochs, test_losses, marker="o", label="Test loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(epochs, test_accs, marker="o", label="Test accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Test Accuracy")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

    filename = f"transformer_{tag}.png"
    save_path = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved plot -> {save_path}")


# Save the trained model weights to disk.
# Writes the model state dictionary to the given path.
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved model -> {path}")


# Run the full Vision Transformer experiment on MNIST.
# Trains, evaluates, saves the best model, and plots results.
def run_transformer(
    epochs=8,
    batch_size=256,
    patch_size=4,
    stride=4,
    embed_dim=64,
    depth=3,
    num_heads=4,
    lr=1e-3,
    weight_decay=1e-4,
    seed=1
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    device = get_device()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    num_workers = get_num_workers()
    train_loader, test_loader = get_data_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
        device=device
    )

    config = NetConfig(
        name="vit_task4",
        patch_size=patch_size,
        stride=stride,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        epochs=epochs,
        device=str(device)
    )

    model = NetTransformer(config).to(device)

    print("\nTransformer architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Batch size: {batch_size}")
    print(f"DataLoader workers: {num_workers}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    train_losses = []
    test_losses = []
    test_accs = []

    best_acc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(MODELS_DIR, "transformer_best.pth")

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            scaler=scaler,
            epoch=epoch
        )

        test_loss, test_acc = evaluate_network(
            model=model,
            loader=test_loader,
            device=device
        )

        scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Test  loss: {test_loss:.4f}")
        print(f"Test  acc : {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            save_model(model, best_model_path)

    tag = f"ps{patch_size}_s{stride}_d{depth}_h{num_heads}"
    plot_results(train_losses, test_losses, test_accs, tag)

    print(f"\nBest test accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    return best_acc, best_epoch


# Parse command-line inputs and launch Task 4.
# Sets experiment configuration and prints final result.
def main(argv):
    epochs = int(argv[1]) if len(argv) > 1 else 8
    patch_size = int(argv[2]) if len(argv) > 2 else 4
    stride = int(argv[3]) if len(argv) > 3 else 4
    embed_dim = int(argv[4]) if len(argv) > 4 else 64
    depth = int(argv[5]) if len(argv) > 5 else 3
    num_heads = int(argv[6]) if len(argv) > 6 else 4
    batch_size = int(argv[7]) if len(argv) > 7 else 256

    print("=" * 60)
    print("  Task 4 – Vision Transformer for MNIST")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  epochs={epochs}")
    print(f"  patch_size={patch_size}")
    print(f"  stride={stride}")
    print(f"  embed_dim={embed_dim}")
    print(f"  depth={depth}")
    print(f"  num_heads={num_heads}")
    print(f"  batch_size={batch_size}")

    best_acc, best_epoch = run_transformer(
        epochs=epochs,
        batch_size=batch_size,
        patch_size=patch_size,
        stride=stride,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads
    )

    print(f"\nTask 4 complete. Best test acc: {best_acc:.2f}% (epoch {best_epoch})")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Unable to import Axes3D")
    main(sys.argv)
