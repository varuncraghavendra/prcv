/*

  Varun Raghavendra

  Spring 2026

  CS 5330 Computer Vision

  Transfer learning on MNIST CNN for Greek letter classification

*/

import sys
import os
import shutil
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

from task1 import MyNetwork

OUTPUTS = 'outputs/task3'
MODELS  = 'saved_models'
os.makedirs(OUTPUTS, exist_ok=True)

LABEL_MAP = {0: 'alpha', 1: 'beta', 2: 'gamma'}


# Transform Greek letter images into MNIST-style tensors.
# Converts to grayscale, rescales, crops, and inverts the image.
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# Build a DataLoader from alpha, beta, and gamma folders.
# Verifies that only the expected three training classes exist.
def get_greek_loader(folder_path, batch_size=5, shuffle=True):
    found = sorted([
        d for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
        and d in ('alpha', 'beta', 'gamma')
    ])
    if found != ['alpha', 'beta', 'gamma']:
        raise RuntimeError(
            f'Expected subfolders alpha/, beta/, gamma/ in {folder_path}\n'
            f'Found: {found}\n'
            'Make sure the training folder contains ONLY these three subfolders.')

    dataset = torchvision.datasets.ImageFolder(
        folder_path,
        transform=transforms.Compose([
            transforms.ToTensor(),
            GreekTransform(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    print(f'  Classes detected: {dataset.classes}')
    print(f'  Class-to-index:   {dataset.class_to_idx}')
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)


# Copy training class folders into a temporary clean directory.
# Prevents extra custom folders from being treated as dataset classes.
def make_training_dir(src_dir, tmp_dir='tmp_greek_train'):
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    for letter in ('alpha', 'beta', 'gamma'):
        src = os.path.join(src_dir, letter)
        dst = os.path.join(tmp_dir, letter)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            raise RuntimeError(f'Required subfolder not found: {src}')
    return tmp_dir


# Load pretrained MNIST weights and adapt model for 3-class output.
# Freezes old layers and replaces the last layer with a trainable head.
def build_greek_model(pretrained_path):
    model = MyNetwork()
    model.load_state_dict(
        torch.load(pretrained_path, map_location='cpu', weights_only=True))
    print(f'  Loaded pretrained weights from {pretrained_path}')

    for param in model.parameters():
        param.requires_grad = False

    model.fc2 = nn.Linear(50, 3)

    print('\nModified network:')
    print(model)
    print()
    for name, param in model.named_parameters():
        print(f'  {name:30s}  requires_grad={param.requires_grad}')

    return model


# Train the new output layer on Greek letter data.
# Stops early if perfect accuracy is maintained for several epochs.
def train_greek(model, loader, max_epochs=100, min_perfect=5):
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    losses, accs   = [], []
    perfect_streak = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss   = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(data)
            correct    += output.argmax(1).eq(target).sum().item()
            total      += len(data)

        avg_loss = epoch_loss / total
        acc      = 100.0 * correct / total
        losses.append(avg_loss)
        accs.append(acc)
        print(f'  Epoch {epoch:3d}  loss: {avg_loss:.4f}  acc: {acc:.1f}%')

        if acc >= 100.0:
            perfect_streak += 1
            if perfect_streak >= min_perfect:
                print(f'  → 100% accuracy held for {min_perfect} consecutive '
                      f'epochs. Stopping at epoch {epoch}.')
                break
        else:
            perfect_streak = 0

    return losses, accs


# Plot training loss and accuracy across epochs.
# Saves the training curves to the task3 output folder.
def plot_training(losses, accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(losses) + 1)

    ax1.plot(epochs, losses, 'b-o', markersize=4)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('NLL Loss')
    ax1.set_title('Training Loss'); ax1.grid(True)

    ax2.plot(epochs, accs, 'g-o', markersize=4)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.set_ylim([0, 105]); ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(OUTPUTS, 'greek_training.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved -> {path}')


# Evaluate the model on every sample in the loader.
# Prints true labels, predictions, and returns accuracy counts.
def evaluate_all(model, loader):
    model.eval()
    correct, total = 0, 0

    print('\n  ── Full dataset evaluation ──')
    print(f'  {"True":<10}  {"Pred":<10}  OK')
    print('  ' + '─' * 28)

    with torch.no_grad():
        for data, targets in loader:
            preds = model(data).argmax(1)
            for t, p in zip(targets, preds):
                t_name = LABEL_MAP[t.item()]
                p_name = LABEL_MAP[p.item()]
                ok     = '✓' if t == p else '✗'
                print(f'  {t_name:<10}  {p_name:<10}  {ok}')
                correct += int(t == p)
                total   += 1

    print(f'\n  Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)')
    return correct, total


# Test the trained model on custom hand-drawn Greek letter images.
# Loads custom folders, predicts labels, and saves a result grid.
def test_custom_images(model, custom_base_dir):
    custom_folders = {
        'alpha': os.path.join(custom_base_dir, 'custom_alpha'),
        'beta':  os.path.join(custom_base_dir, 'custom_beta'),
        'gamma': os.path.join(custom_base_dir, 'custom_gamma'),
    }

    if not any(os.path.isdir(d) for d in custom_folders.values()):
        print(f'  No custom_alpha/beta/gamma folders found in: {custom_base_dir}')
        return

    greek_tf = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    results        = []
    correct, total = 0, 0

    print(f'\n  {"True":<10}  {"Pred":<10}  OK')
    print('  ' + '─' * 28)

    for true_label, folder in custom_folders.items():
        if not os.path.isdir(folder):
            continue
        fnames = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        for fname in fnames:
            pil_img  = Image.open(os.path.join(folder, fname)).convert('RGB')
            tensor   = greek_tf(pil_img).unsqueeze(0)
            with torch.no_grad():
                pred_idx = model(tensor).argmax(1).item()
            pred_label = LABEL_MAP[pred_idx]
            ok         = pred_label == true_label
            correct   += int(ok)
            total     += 1
            results.append((true_label, pred_label, pil_img, ok))
            print(f'  {true_label:<10}  {pred_label:<10}  {"✓" if ok else "✗"}')

    if not results:
        print('  No images found in custom folders.')
        return

    print(f'\n  Custom accuracy: {correct}/{total} ({100 * correct / total:.1f}%)')

    n     = len(results)
    ncols = min(n, 6)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.2, nrows * 2.6))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, (true_lbl, pred_lbl, pil_img, ok) in enumerate(results):
        processed = greek_tf(pil_img).squeeze().numpy()
        lo, hi    = processed.min(), processed.max()
        disp      = (processed - lo) / (hi - lo + 1e-8)
        axes[i].imshow(disp, cmap='gray')
        axes[i].set_title(
            f'True: {true_lbl}\nPred: {pred_lbl}',
            fontsize=8, color='green' if ok else 'red')
        axes[i].axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.suptitle(
        f'Custom Greek Letter Predictions  ({correct}/{total} correct)',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUTS, 'custom_greek_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {path}')


# Run the full transfer learning pipeline for Greek letters.
# Loads data, trains the model, evaluates, tests custom images, and cleans up.
def main(argv):
    greek_dir  = argv[1] if len(argv) > 1 else 'greek_letters'
    max_epochs = int(argv[2]) if len(argv) > 2 else 100
    custom_dir = argv[3] if len(argv) > 3 else greek_dir
    model_path = os.path.join(MODELS, 'mnist_model.pth')

    print('=' * 60)
    print('  Task 3 – Transfer Learning on Greek Letters')
    print('=' * 60)

    if not os.path.exists(model_path):
        print(f'ERROR: model not found at {model_path}')
        print('       Run task1.py first.')
        sys.exit(1)

    if not os.path.exists(greek_dir):
        print(f'ERROR: folder not found: {greek_dir}')
        sys.exit(1)

    print(f'\n[1] Loading Greek training data from: {greek_dir}')
    tmp_dir = make_training_dir(greek_dir)
    try:
        loader = get_greek_loader(tmp_dir)
        print(f'  Samples : {len(loader.dataset)}')

        print('\n[2] Building transfer learning model...')
        model = build_greek_model(model_path)

        print(f'\n[3] Training for up to {max_epochs} epochs...')
        losses, accs = train_greek(model, loader, max_epochs)

        print('\n[4] Saving training plots...')
        plot_training(losses, accs)
        save_path = os.path.join(MODELS, 'greek_model.pth')
        torch.save(model.state_dict(), save_path)
        print(f'  Model saved -> {save_path}')

        print('\n[5] Evaluating on all training samples...')
        evaluate_all(model, loader)

        print(f'\n[6] Testing custom images from: {custom_dir}')
        test_custom_images(model, custom_dir)

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    print('\nTask 3 complete.')


if __name__ == '__main__':
    main(sys.argv)
