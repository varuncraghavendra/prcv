/*

  Varun Raghavendra

  Spring 2026

  CS 5330 Computer Vision

  Systematic hyperparameter sweep for MNIST CNN design experiment

*/

import sys
import os
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from task1 import get_data_loaders, evaluate_network, train_one_epoch

OUTPUTS = 'outputs/task5'
MODELS  = 'saved_models'
os.makedirs(OUTPUTS, exist_ok=True)


# Configurable CNN model for hyperparameter experiments
# Varies conv filters, dropout rate, and hidden layer size
class FlexNet(nn.Module):
    def __init__(self, conv1_filters=10, dropout_rate=0.5, fc_hidden=50):
        super(FlexNet, self).__init__()
        self.conv1_filters = conv1_filters
        conv2_filters      = conv1_filters * 2
        fc_in              = conv2_filters * 4 * 4

        self.conv1   = nn.Conv2d(1, conv1_filters, kernel_size=5)
        self.conv2   = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1     = nn.Linear(fc_in, fc_hidden)
        self.fc2     = nn.Linear(fc_hidden, 10)

    # Forward pass through convolution and fully connected layers
    # Produces log-probability outputs for digit classification
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


# Train and evaluate one hyperparameter configuration
# Returns test accuracy, train accuracy, and runtime
def evaluate_config(conv1_filters, dropout_rate, fc_hidden,
                    epochs=5, batch_size=64, seed=42):
    torch.manual_seed(seed)
    train_loader, test_loader = get_data_loaders(batch_size)

    model     = FlexNet(conv1_filters, dropout_rate, fc_hidden)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    t0 = time.time()
    dummy_counter, dummy_losses = [], []
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, optimizer, train_loader, epoch,
                        dummy_counter, dummy_losses)

    _, test_acc = evaluate_network(model, test_loader)
    _, train_acc = evaluate_network(model, train_loader)
    elapsed = time.time() - t0
    return test_acc, train_acc, elapsed


# Sweep one hyperparameter while fixing the others
# Returns results for all tested values in that dimension
def sweep_dimension(dim_name, values, fixed, epochs=5):
    results = []
    for v in values:
        cfg = dict(fixed)
        cfg[dim_name] = v
        print(f'  {dim_name}={v}  filters={cfg["filters"]}  '
              f'dropout={cfg["dropout"]}  fc_hidden={cfg["fc_hidden"]}  ...',
              end='', flush=True)
        test_acc, train_acc, elapsed = evaluate_config(
            conv1_filters=cfg['filters'],
            dropout_rate=cfg['dropout'],
            fc_hidden=cfg['fc_hidden'],
            epochs=epochs,
        )
        print(f'  test acc={test_acc:.2f}%  ({elapsed:.1f}s)')
        results.append((v, test_acc, train_acc, elapsed))
    return results


# Plot train and test accuracy for a single sweep
# Saves a bar chart for one hyperparameter dimension
def plot_sweep(dim_name, values, test_accs, train_accs, ylabel='Accuracy (%)'):
    x = np.arange(len(values))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, train_accs, w, label='Train acc', color='steelblue', alpha=0.85)
    ax.bar(x + w/2, test_accs,  w, label='Test acc',  color='tomato',    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in values], fontsize=11)
    ax.set_xlabel(dim_name, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Effect of {dim_name}', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([min(min(train_accs), min(test_accs)) - 2, 100])
    plt.tight_layout()
    path = os.path.join(OUTPUTS, f'sweep_{dim_name}.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f'  Saved -> {path}')


# Plot summary of all experiment runs
# Saves a bar chart showing test accuracy for every configuration
def plot_summary(all_results):
    labels = [f"f{r['filters']}_d{r['dropout']}_fc{r['fc_hidden']}"
              for r in all_results]
    accs   = [r['test_acc'] for r in all_results]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
    colors = plt.cm.RdYlGn(
        [(a - min(accs)) / (max(accs) - min(accs) + 1e-9) for a in accs])
    bars = ax.bar(range(len(accs)), accs, color=colors, edgecolor='grey')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('All Experiment Runs – Test Accuracy', fontweight='bold')
    ax.set_ylim([min(accs) - 1, 100.5])
    ax.grid(True, axis='y', alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{acc:.1f}', ha='center', va='bottom', fontsize=6)
    plt.tight_layout()
    path = os.path.join(OUTPUTS, 'all_runs_summary.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f'  Saved -> {path}')


# Save all experiment results into a CSV file
# Writes configuration values, accuracies, time, and round info
def save_csv(all_results):
    path = os.path.join(OUTPUTS, 'results.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['filters', 'dropout', 'fc_hidden',
                           'test_acc', 'train_acc', 'time_s', 'round', 'dim'])
        writer.writeheader()
        writer.writerows(all_results)
    print(f'  CSV saved -> {path}')


# Run the full hyperparameter search experiment
# Sweeps filters, dropout, and hidden size across multiple rounds
def main(argv):
    epochs_per_run = int(argv[1]) if len(argv) > 1 else 5
    rounds         = int(argv[2]) if len(argv) > 2 else 3

    print('=' * 60)
    print('  Task 5 – Design Your Own Experiment')
    print(f'  Epochs/run: {epochs_per_run}  |  Rounds: {rounds}')
    print('=' * 60)

    print("""
  Hypotheses (before running):
    H1 [filters]:    More filters → better accuracy up to ~24 filters,
                     then diminishing returns and slower training.
    H2 [dropout]:    Moderate dropout (0.25-0.5) will outperform extremes.
                     Low dropout (<0.1) will overfit; high dropout (>0.6)
                     will underfit.
    H3 [fc_hidden]:  Larger FC layer improves accuracy modestly;
                     100 nodes likely optimal for this task.
    """)

    filters_values  = [10, 16, 24, 32]
    dropout_values  = [0.1, 0.25, 0.5, 0.7]
    fc_hidden_values = [32, 50, 100, 200]

    best = {'filters': 10, 'dropout': 0.5, 'fc_hidden': 50}
    all_results = []

    for rnd in range(1, rounds + 1):
        print(f'\n{"─"*60}')
        print(f'  Round {rnd}  (current best: {best})')
        print('─' * 60)

        print('\n  [D1] Sweeping number of conv1 filters...')
        r1 = sweep_dimension('filters', filters_values, best, epochs_per_run)
        for v, ta, tra, t in r1:
            all_results.append({'filters': v, 'dropout': best['dropout'],
                                 'fc_hidden': best['fc_hidden'],
                                 'test_acc': ta, 'train_acc': tra,
                                 'time_s': round(t, 1), 'round': rnd, 'dim': 'filters'})
        best_v = max(r1, key=lambda x: x[1])[0]
        best['filters'] = best_v
        print(f'  → Best filters: {best_v}')
        plot_sweep('filters', filters_values,
                   [x[1] for x in r1], [x[2] for x in r1])

        print('\n  [D2] Sweeping dropout rate...')
        r2 = sweep_dimension('dropout', dropout_values, best, epochs_per_run)
        for v, ta, tra, t in r2:
            all_results.append({'filters': best['filters'], 'dropout': v,
                                 'fc_hidden': best['fc_hidden'],
                                 'test_acc': ta, 'train_acc': tra,
                                 'time_s': round(t, 1), 'round': rnd, 'dim': 'dropout'})
        best_v = max(r2, key=lambda x: x[1])[0]
        best['dropout'] = best_v
        print(f'  → Best dropout: {best_v}')
        plot_sweep('dropout', dropout_values,
                   [x[1] for x in r2], [x[2] for x in r2])

        print('\n  [D3] Sweeping FC hidden layer size...')
        r3 = sweep_dimension('fc_hidden', fc_hidden_values, best, epochs_per_run)
        for v, ta, tra, t in r3:
            all_results.append({'filters': best['filters'],
                                 'dropout': best['dropout'], 'fc_hidden': v,
                                 'test_acc': ta, 'train_acc': tra,
                                 'time_s': round(t, 1), 'round': rnd, 'dim': 'fc_hidden'})
        best_v = max(r3, key=lambda x: x[1])[0]
        best['fc_hidden'] = best_v
        print(f'  → Best fc_hidden: {best_v}')
        plot_sweep('fc_hidden', fc_hidden_values,
                   [x[1] for x in r3], [x[2] for x in r3])

        print(f'\n  End of round {rnd}.  Updated best: {best}')

    print('\n  Saving summary...')
    plot_summary(all_results)
    save_csv(all_results)

    overall_best = max(all_results, key=lambda x: x['test_acc'])
    print(f"""
  ── Final Results ──────────────────────────────────────
  Best config:
    conv1_filters = {overall_best['filters']}
    dropout       = {overall_best['dropout']}
    fc_hidden     = {overall_best['fc_hidden']}
    Test accuracy = {overall_best['test_acc']:.2f}%

  Hypothesis validation:
    H1 (filters):   {'SUPPORTED' if overall_best['filters'] > 10 else 'NOT SUPPORTED'}
    H2 (dropout):   check sweep_dropout.png
    H3 (fc_hidden): {'SUPPORTED' if overall_best['fc_hidden'] >= 100 else 'PARTIAL'}
  ───────────────────────────────────────────────────────
    """)

    print('Task 5 complete.')


if __name__ == '__main__':
    main(sys.argv)
