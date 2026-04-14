/*

  Varun Raghavendra

  Spring 2026

  CS 5330 Computer Vision

  Analyze trained CNN filters and visualize their effect on MNIST images

*/

import os
import sys
import warnings

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from task1 import MyNetwork

OUTPUTS_DIR = "outputs/task2"
DEFAULT_MODEL_PATH = "saved_models/mnist_model.pth"
DATA_DIR = "data"

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# Convert a PyTorch tensor into a NumPy array safely.
# Detaches from graph and moves data to CPU before conversion.
def safe_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


# Load the trained MNIST model from disk.
# Restores saved weights and returns the model in eval mode.
def load_model(model_path: str = DEFAULT_MODEL_PATH) -> MyNetwork:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run task1.py first to train and save the model."
        )

    model = MyNetwork()

    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        warnings.warn(
            "Your torch version does not support weights_only=True. "
            "Falling back to standard torch.load(...).",
            UserWarning,
        )
        state_dict = torch.load(model_path, map_location="cpu")

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded from {model_path}")
    print("\nModel structure:")
    print(model)

    return model


# Load the first MNIST training image in raw and normalized forms.
# Returns raw image, normalized image, and the corresponding label.
def get_mnist_images():
    raw_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    raw_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=raw_transform,
    )

    norm_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=norm_transform,
    )

    raw_image_tensor, raw_label = raw_dataset[0]
    norm_image_tensor, _ = norm_dataset[0]

    raw_image = safe_tensor_to_numpy(raw_image_tensor.squeeze(0))
    norm_image = safe_tensor_to_numpy(norm_image_tensor.squeeze(0))

    return raw_image, norm_image, raw_label


# Extract and visualize the first convolution layer filters.
# Prints filter weights and saves a plot of all conv1 kernels.
def analyze_first_layer(model: MyNetwork) -> np.ndarray:
    weights = safe_tensor_to_numpy(model.conv1.weight)  # shape: [10, 1, 5, 5]

    print(f"\nconv1 weight shape: {weights.shape}")
    print("  [num_filters, in_channels, H, W] =", list(weights.shape))
    print()

    for i in range(weights.shape[0]):
        filt = weights[i, 0]
        print(f"  Filter {i}:")
        print(filt)
        print()

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    fig.suptitle("Conv1 Filter Weights (10 filters, 5x5)", fontsize=13, fontweight="bold")

    flat_axes = axes.flatten()
    for i in range(10):
        ax = flat_axes[i]
        filt = weights[i, 0]
        im = ax.imshow(filt, cmap="viridis")
        ax.set_title(f"Filter {i}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(10, 12):
        flat_axes[j].axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_DIR, "conv1_filters.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {save_path}")
    return weights


# Apply each conv1 filter to the first MNIST image.
# Saves a visualization of original and filtered outputs.
def apply_filters_to_image(model: MyNetwork) -> None:
    raw_image, norm_image, raw_label = get_mnist_images()
    weights = safe_tensor_to_numpy(model.conv1.weight)  # [10, 1, 5, 5]

    rows, cols = 10, 4
    fig, axes = plt.subplots(rows, cols, figsize=(10, 22))

    column_titles = ["Filter", "Original", "Filtered (inverted)", "Filtered"]
    for c, title in enumerate(column_titles):
        axes[0, c].set_title(title, fontsize=10, fontweight="bold")

    for i in range(10):
        filt = weights[i, 0].astype(np.float32)
        filtered = cv2.filter2D(norm_image.astype(np.float32), ddepth=-1, kernel=filt)

        axes[i, 0].imshow(filt, cmap="viridis")
        axes[i, 0].set_ylabel(f"Filter {i}", fontsize=8)

        axes[i, 1].imshow(raw_image, cmap="gray")
        axes[i, 1].set_xlabel(f"Digit: {raw_label}", fontsize=8)

        axes[i, 2].imshow(filtered, cmap="gray_r")
        axes[i, 3].imshow(filtered, cmap="gray")

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_DIR, "filtered_images.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {save_path}")


# Run the full Task 2 analysis pipeline.
# Loads model, visualizes filters, and applies them to an image.
def main(argv):
    model_path = argv[1] if len(argv) > 1 else DEFAULT_MODEL_PATH

    print("=" * 60)
    print("  Task 2 - Network Analysis")
    print("=" * 60)

    try:
        model = load_model(model_path)

        print("\n[1] Analyzing conv1 filters...")
        analyze_first_layer(model)

        print("\n[2] Applying filters to first training image...")
        apply_filters_to_image(model)

        print("\nTask 2 complete.")

    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
