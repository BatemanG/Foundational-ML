"""
loss.py

Contains the loss functions for XGBoost.
"""

import numpy as np
from typing import Tuple , Protocol
from dataclasses import dataclass

# ===================================================================================
#  Loss function
# ===================================================================================

@dataclass(frozen=True)
class MultiClassLogLoss:
    '''The log loss for multi-class classification '''
    def compute_gh(self, y_true_one_hot: np.ndarray, y_pred_logit: np.ndarray) -> Tuple[float, float]:
        """
        Vectoirsed Computation of gradient and hessian.

        Args:
            y_true_one_hot: (n, c) array of one-hot encoded labels
            y_pred_logit: Raw logit predictions (n, c) array

        Returns:
            (g, h): Gradient an dHessian Matrices (n_samples, n_classes)
        """
        # Softmax calculation
        p = self._softmax(y_pred_logit)

        g = p - y_true_one_hot
        h = p * (1.0 - p)
        return g, h

    @staticmethod
    def _softmax(y_pred_logit: np.ndarray) -> np.ndarray:
        """Vectorized Stable softmax calculation"""
        exp_logits = np.exp(y_pred_logit - np.max(y_pred_logit, axis=1, keepdims=True))  #Note: we take away the max to keep numbers small
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True) # (N,K)
