"""
_base.py

Contains the base dataclasses, protocols, and type definitions
used across the quant_ml_numpy package.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Protocol
import numpy as np

# ===================================================================================
#  Core Data Structures and Configuration
# ===================================================================================

@dataclass(frozen=True)
class BoosterConfig:
    '''Hyperparameters for the boosting process'''
    n_estimators: int = 100
    learning_rate: float = 0.1

@dataclass(frozen=True)
class SplitConfig:
    '''Hyperparameters for the split process'''
    gamma: float = 0.0
    lambda_:float = 1.0
    min_hessian: float = 1
    max_depth: int = 3
    algorithm: str = "goss"
    # GOSS parameters
    goss_a: float = 0.2 # The ration of large gradinet instances
    goss_b: float = 0.1 # The ration of small gradient instance


@dataclass(frozen=True)
class SplitInfo:
    '''Stores the optimal split found for a node.'''
    feature_idx: int
    threshold: float
    gain: float
    g_left: float
    g_right: float
    h_left: float
    h_right: float

@dataclass
class Node:
    ''' A recursive data structure representing a node in the decision tree.'''
    depth: int
    instance_indices: np.ndarray

    split_info: Optional[SplitInfo] = None
    left_child: Optional['Node'] = None
    right_child: Optional['Node'] = None
    weight: Optional[float] = None

    @property
    def is_leaf(self) -> bool:
        return self.left_child is None and self.right_child is None


class Loss(Protocol):
    def compute_gh(self, y_true: np.ndarray, y_pred_logit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...
