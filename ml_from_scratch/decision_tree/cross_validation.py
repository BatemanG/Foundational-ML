import numpy as np
from dataclasses import dataclass
from .metrics import confusion_matrix, accuracy_from_confusion, recall_precision_f1_per_class
from .tree import DecisionTree
from .config import DecisionTreeConfig
from .utils import test_train_k_fold, get_max_depth


@dataclass(frozen=True)
class CrossValidationResult:
    confusion: np.ndarray
    accuracy: float
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    macro_precision: float
    macro_recall: float
    macro_f1: float
    average_tree_depth: float

    def __str__(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.3f}\n"
            f"Macro Precision: {self.macro_precision:.3f}\n"
            f"Macro Recall: {self.macro_recall:.3f}\n"
            f"Macro F1: {self.macro_f1:.3f}\n"
            f"Confusion Matrix:\n{self.confusion}\n"
        )

    def get_average_tree_depth(self) -> str:
        return f"Average Tree Depth: {self.average_tree_depth:.3f}"


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
    gain_threshold: float = 0.0,
    max_depth: int = 50,
    min_samples_split: int = 2,
    random_state: int = 42,
    prune_tree: bool = False,
    alpha: float = 0.0,
    tol: float = 0.0,
) -> CrossValidationResult:
    """Perform deterministic k-fold cross-validation for DecisionTree.

    If `prune_tree=True`, performs nested CV: each outer test fold is evaluated
    by training and pruning across all (n_folds - 1) inner splits.
    Otherwise, standard n-fold CV is used.

    Args:
        X: (n, d) float features
        y: (n,) integer labels
        n_folds: number of cross-validation folds
        max_depth, min_samples_split, gain_threshold: DecisionTree hyperparams
        random_state: RNG seed

    Returns:
        Cross_Validation_Result dataclass with aggregated metrics.
    """
    assert n_folds >= 2, "n_folds must be at least 2"
    assert len(X) >= n_folds, "n_folds must be less than or equal to the number of instances"

    n_classes = len(np.unique(y))
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

    total_depth = 0
    num_trees = 0  # counter for average updating

    config = DecisionTreeConfig(max_depth, min_samples_split, gain_threshold, alpha, tol)

    rng = np.random.default_rng(random_state)

    # test folds
    for test_idx, train_idx in test_train_k_fold(n_folds, X, rng):
        X_test, y_test = X[test_idx], y[test_idx]
        X_train, y_train = X[train_idx], y[train_idx]

        if prune_tree:
            # Nested CV for prunning evaluation
            for val_idx, train_idx_inner in test_train_k_fold(n_folds - 1, X_train, rng):
                X_val, y_val = X_train[val_idx], y_train[val_idx]
                X_train_inner, y_train_inner = X_train[train_idx_inner], y_train[train_idx_inner]
                tree = train_tree(
                    X_train_inner, y_train_inner, X_val, y_val, config=config, prune=prune_tree
                )
                confusion += evaluate_tree(tree, X_test, y_test)
                total_depth += get_max_depth(tree.root) - 1
                num_trees += 1

        else:
            tree = train_tree(X_train, y_train, config=config, prune=prune_tree)
            confusion += evaluate_tree(tree, X_test, y_test)
            depth = get_max_depth(tree.root) - 1
            total_depth += get_max_depth(tree.root) - 1
            num_trees += 1

    average_depth = total_depth / num_trees if num_trees > 0 else 0
    return _aggregate_results(confusion, average_depth)


def train_tree(X_train, y_train, X_val=None, y_val=None, *, config, prune=False):
    """Fit a decision tree and optionally prune it."""
    tree = DecisionTree(config).fit(X_train, y_train)
    if prune and X_val is not None and y_val is not None:
        tree.prune(X_train, y_train, X_val, y_val)
    return tree


def evaluate_tree(tree, X_test, y_test):
    """Compute Confusion matrix for a tree (a single fold)"""
    y_prediction = tree.predict(X_test)
    return confusion_matrix(y_test, y_prediction)


def _aggregate_results(confusion: np.ndarray, average_tree_depth) -> CrossValidationResult:
    accuracy = accuracy_from_confusion(confusion)
    recall, precision, f1 = recall_precision_f1_per_class(confusion)
    macro_r, macro_p, macro_f = map(np.mean, (recall, precision, f1))

    return CrossValidationResult(
        confusion=confusion,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f,
        average_tree_depth=average_tree_depth,
    )

