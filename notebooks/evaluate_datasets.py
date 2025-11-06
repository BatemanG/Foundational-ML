from typing import Tuple
import numpy as np

import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

from ml_from_scratch.decision_tree.cross_validation import cross_validate


# Load Data
def load_wifi_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    X = arr[:, :-1].astype(np.float64)
    y_raw = arr[:, -1].astype(np.int64)
    y = y_raw - 1  # Since we want to start with 0
    return X, y


def load_dataset(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for 'clean' or 'noisy' dataset."""
    if dataset == "clean":
        return load_wifi_txt("data/wifi_db/clean_dataset.txt")
    elif dataset == "noisy":
        return load_wifi_txt("data/wifi_db/noisy_dataset.txt")
    else:
        raise ValueError("dataset must be 'clean' or 'noisy'")


def evaluate_tree_dataset(
    dataset: str = "clean",  # "clean" or "noisy"
    prune_tree: bool = True,  # whether to apply pruning
    max_depth: int = 50,
    n_folds: int = 5,
    gain_threshold: float = 0.0,
    alpha: float = 0.0,
    tol: float = 0.0,
) -> None:
    """Evaluate DecisionTree with or without pruning on clean or noisy dataset.

    Args:
        dataset: "clean" or "noisy"
        prune_tree: whether to apply pruning
        max_depth: maximum depth of the tree
        n_folds: number of folds for cross-validation
    """
    # --- Select dataset ---
    X, y = load_dataset(dataset)

    # --- Run cross-validation ---
    results = cross_validate(
        X=X,
        y=y,
        n_folds=n_folds,
        max_depth=max_depth,
        prune_tree=prune_tree,
        gain_threshold=gain_threshold,
        alpha=alpha,
        tol=tol,
    )

    # --- Print summary ---
    title = (
        f"Results on {dataset.title()} dataset {'(Pruned)' if prune_tree else '(Unpruned)'} "
        f"max_depth={max_depth}, folds={n_folds}"
    )

    print("=" * len(title))
    print(title)
    print("=" * len(title))
    print(results)
    print(results.get_average_tree_depth())
