from dataclasses import dataclass
import numpy as np
from .split import Split

@dataclass()
class Node:
    left: 'Node | None' = None
    right: 'Node | None' = None
    split: Split | None = None
    feature: int | None = None
    threshold: float | None = None
    label: int | None = None # Label if leaf node
    depth: int = 0

    # --- For pruning ----
    train_idx: np.ndarray | None = None        # Label prunned leafs
    val_idx: np.ndarray | None = None          # Working out what to prun

    @property
    def is_leaf(self) -> bool:
        return self.label is not None

    def predict(self, x: np.ndarray) -> int:
        if x.ndim > 1: # Ensure x is a 1D row vector
            x = x.ravel()
        node = self
        while not node.is_leaf:
            j = node.feature
            t = node.threshold
            node = node.left if x[j] < t else node.right
        return int(node.label)