import numpy as np
from numpy.random import default_rng
from dataclasses import dataclass
from typing import Tuple
from ._base import BoosterConfig, SplitConfig, SplitInfo, Node, Loss
from .loss import MultiClassLogLoss

# ---------------------------------------------------------------------
# Cross-validation result dataclass
# ---------------------------------------------------------------------
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
    # average_tree_depth: float

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

# ---------------------------------------------------------------------
# Metrics utilities (self-contained)
# ---------------------------------------------------------------------
def confusion_matrix(y_correct, y_prediction, class_labels=None):
    """ Compute the confusion matrix.

    Args:
        y_correct (np.ndarray): the correct ground truth/gold standard labels (N,)
        y_prediction (np.ndarray): the predicted labels   (M,)
        class_labels (np.ndarray): a list of unique class labels.
                               Defaults to the union of y_correct and y_prediction.

    Returns:
        np.array : shape (C, C), where C is the number of classes.
                   Rows are ground truth per class, columns are predictions
    """
    if class_labels is None:
        class_lables = np.unique(np.concatenate((y_correct, y_prediction)))

    # Vectoised with One-hot encoding
    y_correct_one_hot = (y_correct[..., np.newaxis] == class_lables).astype(np.int64) # (N,) -> (N, 1) -> (N, C)
    y_prediction_one_hot = (y_prediction[..., np.newaxis] == class_lables).astype(np.int64) #(M,) -> (M,1) -> (M,C)

    return y_correct_one_hot.T @ y_prediction_one_hot # (C, N) @ (N, C) -> (C, C)

def accuracy_from_confusion(confusion):
    """ Compute the accuracy given the confusion matrix

    Args:
        confusion (np.ndarray): shape (C, C), where C is the number of classes.
                    Rows are ground truth per class, columns are predictions

    Returns:
        float : the accuracy
    """

    if np.sum(confusion) > 0:
        return np.trace(confusion) / np.sum(confusion)
    else:
        return 0.

def accuracy(y_correct, y_prediction):
    """ Compute the accuracy given the ground truth and predictions

    Args:
        y_correct (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        float : the accuracy
    """

    assert len(y_correct) == len(y_prediction)
    try:
        # return np.sum(y_correct == y_prediction) / len(y_correct)
        return np.mean(y_correct == y_prediction)
    except ZeroDivisionError:
        return 0.

def recall_precision_f1_per_class(confusion_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the recall, precision, and F1-score per class given a confusion matrix."""
    true_positives = np.diagonal(confusion_matrix)
    actual_positives = np.sum(confusion_matrix, axis=1)
    predicted_positives = np.sum(confusion_matrix, axis=0)
    precision = np.divide(true_positives, predicted_positives,
                          out=np.zeros_like(predicted_positives, dtype=float),
                          where=predicted_positives > 0)
    recall = np.divide(true_positives, actual_positives,
                       out=np.zeros_like(actual_positives, dtype=float),
                       where=actual_positives > 0)
    f1_score = np.divide(2 * precision * recall, precision + recall,
                         out=np.zeros_like(true_positives, dtype=float),
                         where=precision + recall > 0)
    return recall, precision, f1_score

# ---------------------------------------------------------------------
# Deterministic K-fold splitting
# ---------------------------------------------------------------------
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

    for test_idx in split_indices:
        train_idx = np.setdiff1d(all_indices, test_idx)
        yield train_idx, test_idx

# ---------------------------------------------------------------------
# Main cross-validation for numpy-based GBT
# ---------------------------------------------------------------------
def cross_validate_xgb(
    X: np.ndarray,
    y: np.ndarray,
    GBT_class,
    booster_config: BoosterConfig,
    split_config: SplitConfig,
    loss: Loss = MultiClassLogLoss(),
    n_folds: int = 10,
    max_depth: int = 6,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> CrossValidationResult:
    """
    Self-contained k-fold cross-validation for a numpy-based GBT (XGBoost-like) model.
    Produces the same metrics format as the DecisionTree cross-validation.

    Args:
        X, y : features and labels
        GBT_class : class implementing your Gradient Boosted Trees (must have .fit() and .predict())
    """
    n_classes = len(np.unique(y))
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    total_depth = 0
    num_models = 0

    rng = np.random.default_rng(random_state)

    for test_idx, train_idx in test_train_k_fold(n_folds, X, rng):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Instantiate and train your model
        model = GBT_class(
            booster_config=booster_config,
            split_config=split_config,
            loss=loss,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        confusion += confusion_matrix(y_test, y_pred)

    return _aggregate_results(confusion)

def _aggregate_results(confusion: np.ndarray) -> CrossValidationResult:
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
    )