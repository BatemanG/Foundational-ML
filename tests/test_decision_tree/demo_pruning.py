import numpy as np
import copy
import os
import sys


from ml_from_scratch.decision_tree.plotting import _plot_recursive
from ml_from_scratch.decision_tree.utils import k_fold_split
from ml_from_scratch.decision_tree.tree import DecisionTree
from ml_from_scratch.decision_tree.config import DecisionTreeConfig
from ml_from_scratch.decision_tree.cross_validation import train_tree, evaluate_tree

from tests.test_decision_tree.evaluate_datasets import load_dataset


def plot_tree_pair(before_root, after_root, max_depth, figsize=(30, 9)):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(2 * figsize[0], figsize[1]))

    for ax, root, title in zip(
        axes, [before_root, after_root], ["Before Pruning", "After Pruning"]
    ):
        _plot_recursive(ax, root, max_depth, x=0, y=0, h_spread=2**max_depth, v_dist=2.0)
        ax.axis("off")
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    print('=== Decision Tree Evaluation ===')
    dataset = input('Dataset (clean/noisy): ').strip().lower() or 'clean'
    max_depth = int(input('Maximum depth: '))
    n_folds = int(input('Number of folds: '))

    X, y = load_dataset('noisy')

    # Split
    k = 10
    split_indcies = k_fold_split(k, len(X))
    test_idx, val_idx = split_indcies[0], split_indcies[1]
    train_idx = np.setdiff1d(np.arange(len(X)), np.concatenate([test_idx, val_idx]))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Fit Tree
    config = DecisionTreeConfig(max_depth=max_depth)
    tree = train_tree(X_train, y_train, X_val=X_val, y_val=y_val, config=config, prune=False)

    before_root = copy.deepcopy(tree.root)
    # Plot pre pruning
    print(f'Before pruning:\n{evaluate_tree(tree, X_test, y_test)}')

    # Prune
    tree.prune(X_train, y_train, X_val, y_val)
    print(f'After pruning:\n{evaluate_tree(tree, X_test, y_test)}')

    plot_tree_pair(before_root, tree.root, max_depth)



