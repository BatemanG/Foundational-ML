import numpy as np
from typing import Tuple


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