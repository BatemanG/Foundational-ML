from numpy.random import default_rng
import numpy as np
from .node import Node


def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    """Split n_instances into n mutually exclusive splits at random.

    Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list (length n_splits). Each element in the list should contain a
            numpy array giving the indices of the instances in that split.
    """
    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices


def test_train_k_fold(n_folds, X, random_generator=default_rng()):
    """Generate train and test indices at each fold.

    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        Itterable: a list of length n_folds. Each element in the list is a list (or tuple)
            with two elements: a numpy array containing the train indices, and another
            numpy array containing the test indices.
    """
    split_indices = k_fold_split(n_folds, len(X), random_generator=random_generator)
    all_indices = np.arange(len(X))
    return (
        (test_idx, train_idx := np.setdiff1d(all_indices, test_idx)) for test_idx in split_indices
    )


def test_val_train_k_fold(n_folds, X, random_generator=default_rng()):
    """Generate train, val, and test indices at each fold."""
    if n_folds < 3:
        raise ValueError("n_folds must be at least 3 for train, val, and test splits")

    split_indices = k_fold_split(n_folds, len(X), random_generator=random_generator)
    n = len(split_indices)
    all_indices = np.arange(len(X))
    return (
        (
            test := split_indices[i],
            val := split_indices[(i + 1) % n],
            np.setdiff1d(all_indices, np.concatenate([test, val])),
        )
        for i in range(n)
    )


def make_folds(
    X: np.ndarray, n_folds: int, use_validation: bool, random_generator: np.random.Generator
) -> list[np.ndarray]:
    """Always returns (test, val, train)"""
    if use_validation:
        yield from test_val_train_k_fold(n_folds, X, random_generator=random_generator)
    else:
        for test_idx, train_idx in test_train_k_fold(n_folds, X, random_generator=random_generator):
            yield test_idx, np.array([], dtype=np.int64), train_idx


def get_max_depth(root: Node | None) -> int:
    """Return the maximal depth (height) of the tree."""
    # Base case: An empty tree has a depth of 0
    if not root:
        return 0

    # Recursively find the depth of the left and right subtrees
    left_depth = get_max_depth(root.left)
    right_depth = get_max_depth(root.right)

    # The depth of this node is 1 (for itself) + the max of its children
    return 1 + max(left_depth, right_depth)

