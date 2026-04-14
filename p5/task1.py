/*

  Varun Raghavendra

  Spring 2026

  CS 5330 Computer Vision

  CNN training and handwritten digit recognition on MNIST

*/

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUTPUTS  = 'outputs/task1'
MODELS   = 'saved_models'
DATA_DIR = 'data'
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(MODELS,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


class MyNetwork(nn.Module):
    # CNN model for MNIST classification
    # Defines conv + fc layers and forward pass
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1   = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2   = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1     = nn.Linear(320, 50)
        self.fc2     = nn.Linear(50, 10)

    # Forward pass through CNN layers
    # Applies conv, pooling, flattening and classification
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


# Load MNIST dataset and create dataloaders
# Returns train and test loaders with normalization
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(root=DATA_DIR, train=True,
                                           download=True, transform=transform)
    test_set  = torchvision.datasets.MNIST(root=DATA_DIR, train=False,
                                           download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size,
                                               shuffle=False)
    return train_loader, test_loader


# Plot first 6 test images from MNIST
# Saves visualization to output folder
def plot_first_six(test_loader):
    images, labels = next(iter(test_loader))
    fig, axes = plt.subplots(1, 6, figsize=(14, 3))
    fig.suptitle('First Six MNIST Test Digits', fontsize=14, fontweight='bold')
    for i in range(6):
        axes[i].imshow(images[i].squeeze().numpy(), cmap='gray')
        axes[i].set_title(f'Label: {labels[i].item()}', fontsize=11)
        axes[i].axis('off')
    plt.tight_layout()
    path = os.path.join(OUTPUTS, 'first_six_digits.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {path}')


# Plot training loss per batch and test loss per epoch
# Saves combined loss curve visualization
def plot_batch_loss_curve(train_counter, train_losses_batch,
                          test_counter, test_losses):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_counter, train_losses_batch,
            color='steelblue', linewidth=0.8, alpha=0.85, label='Train loss')
    ax.scatter(test_counter, test_losses, color='red', s=60,
               zorder=5, label='Test loss (epoch)')
    ax.set_xlabel('Number of training examples seen')
    ax.set_ylabel('Negative log likelihood loss')
    ax.set_title('Training & Test Loss')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    path = os.path.join(OUTPUTS, 'loss_curve.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f'  Saved -> {path}')


# Plot train and test accuracy across epochs
# Saves accuracy curve visualization
def plot_accuracy_curves(epochs_range, train_accs, test_accs):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, train_accs, 'b-o', label='Train accuracy')
    ax.plot(epochs_range, test_accs, 'r-o', label='Test accuracy')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training & Testing Accuracy')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    path = os.path.join(OUTPUTS, 'accuracy_curves.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f'  Saved -> {path}')


# Train model for one epoch
# Updates weights and tracks batch losses
def train_one_epoch(model, optimizer, train_loader, epoch,
                    train_counter, train_losses_batch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss   = F.nll_loss(output, target)
        loss.backward(); optimizer.step()
        seen = (batch_idx + 1) * len(data) + (epoch - 1) * len(train_loader.dataset)
        train_counter.append(seen)
        train_losses_batch.append(loss.item())


# Evaluate model on dataset
# Returns average loss and accuracy
def evaluate_network(model, loader):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, target in loader:
            output      = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            correct    += output.argmax(1).eq(target).sum().item()
    n = len(loader.dataset)
    return total_loss / n, 100.0 * correct / n


# Save trained model weights
# Stores model in saved_models folder
def save_model(model):
    path = os.path.join(MODELS, 'mnist_model.pth')
    torch.save(model.state_dict(), path)
    print(f'  Model saved -> {path}')


# Evaluate first 10 test samples
# Prints predictions and plots first 9 digits
def evaluate_first_ten(model, test_loader):
    model.eval()
    images, labels = next(iter(test_loader))
    images10 = images[:10]; labels10 = labels[:10]

    predictions = []
    with torch.no_grad():
        for i in range(10):
            out  = model(images10[i].unsqueeze(0)).squeeze()
            pred = out.argmax().item()
            predictions.append(pred)

    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images10[i].squeeze().numpy(), cmap='gray')
        ax.set_title(f'{predictions[i]}')
        ax.axis('off')

    path = os.path.join(OUTPUTS, 'first_9_predictions.png')
    plt.savefig(path); plt.close()


# Detect digit bounding boxes using OpenCV
# Uses HSV thresholding and contour filtering
def detect_digit_boxes(image_path):
    import cv2

    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, sat_thresh = cv2.threshold(sat, 40, 255, cv2.THRESH_BINARY)

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (22, 22))
    dilated = cv2.dilate(sat_thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w * h > 20000):
            boxes.append((x, y, w, h))

    return boxes, img


# Crop and preprocess digit image
# Converts to MNIST format tensor
def crop_and_preprocess(img_bgr, box):
    import cv2

    x, y, w, h = box
    crop = img_bgr[y:y+h, x:x+w]

    gray   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))

    arr = resized.astype(np.float32) / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    norm   = (arr - 0.1307) / 0.3081
    tensor = torch.tensor(norm).unsqueeze(0).unsqueeze(0)
    return tensor, arr


# Run handwritten digit recognition pipeline
# Detects digits, classifies, and plots results
def run_handwritten_recognition(model, image_path):
    if not os.path.exists(image_path):
        return

    model.eval()
    boxes, img_bgr = detect_digit_boxes(image_path)

    for i, box in enumerate(boxes):
        tensor, _ = crop_and_preprocess(img_bgr, box)
        with torch.no_grad():
            pred = model(tensor).argmax().item()
        print(f'{i}: {pred}')


# Main training and evaluation pipeline
# Handles training, saving, testing and handwritten input
def main(argv):
    epochs        = int(argv[1]) if len(argv) > 1 else 5
    batch_size    = int(argv[2]) if len(argv) > 2 else 64
    hw_image_path = argv[3]      if len(argv) > 3 else 'IMG_0074.jpeg'

    train_loader, test_loader = get_data_loaders(batch_size)

    model     = MyNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_counter, train_losses_batch = [], []

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, optimizer, train_loader, epoch,
                        train_counter, train_losses_batch)

    save_model(model)

    model_eval = MyNetwork()
    model_eval.load_state_dict(
        torch.load(os.path.join(MODELS, 'mnist_model.pth'), map_location='cpu'))

    evaluate_first_ten(model_eval, test_loader)
    run_handwritten_recognition(model_eval, hw_image_path)


if __name__ == '__main__':
    main(sys.argv)
