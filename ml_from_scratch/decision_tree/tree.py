"""Contains all the core logic for the tree building/pruning process."""

import numpy as np
from .node import Node
from .config import DecisionTreeConfig
from .split import Split


class DecisionTree:
    """Greedy, decision tree classifier (entropy / infomration gain)."""

    def __init__(self, config: DecisionTreeConfig = DecisionTreeConfig()) -> None:
        self.config = config
        self.root: Node | None = None
        self._n_classes: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        """Train the decsion tree on data."""
        # Add in Error cases
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X must be (n,d) and y must be (n,)")

        self._n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for a batch."""
        if self.root is None:
            raise ValueError("The tree has not been trained yet")
        # preidctions = np.array([self.root.predict(row) for row in X], dtype=np.int64)
        return np.fromiter((self.root.predict(row) for row in X), dtype=np.int64)

    @staticmethod
    def _label_counts(y: np.ndarray, n_classes: int) -> np.ndarray:
        """Compute integer class counts for labels y.

        Args:
            y: integer labels (shape (n,))
            n_classes: total number of classes
        Returns:
            counts: (n_classes,) int64
        """
        return np.bincount(y, minlength=n_classes)

    @staticmethod
    def _entropy_from_counts(counts: np.ndarray) -> float:
        """Shannon entropy H = -∑ p log₂ p.

        Args:
            counts: class counts (non-negative ints)

        Returns:
            Entropy in bits as float
        """
        n = int(counts.sum())
        if n <= 1:
            return 0
        p = counts[counts > 0].astype(np.float64) / n
        return float(-np.dot(p, np.log2(p)))  # np.dot over sum(p * logp) for speed

    @classmethod
    def _remainder(cls, left: np.ndarray, right: np.ndarray) -> float:
        nL, nR = int(left.sum()), int(right.sum())
        n = nL + nR
        if n == 0:
            return 0.0
        HL = cls._entropy_from_counts(left)
        HR = cls._entropy_from_counts(right)
        return (nL * HL + nR * HR) / n

    @classmethod
    def _information_gain(cls, parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        return cls._entropy_from_counts(parent) - cls._remainder(left, right)

    def _best_threshold_for_feature(
        self, xj: np.ndarray, y: np.ndarray, feature: int, gain_threshold: float = 0.0
    ) -> None | Split:
        n = xj.shape[0]
        if n <= 1:
            return None

        # Sort bt feature value
        order = np.argsort(xj)
        xj_sorted, y_sorted = xj[order], y[order]

        # Start with S_all = S_right, S_left = [] and work right to left
        S_all = self._label_counts(y_sorted, self._n_classes)
        S_R = S_all.copy()
        S_L = np.zeros(self._n_classes, dtype=np.int64)

        best_gain = 0.0
        best_split: float | None = None

        for idx, label in enumerate(y_sorted[:-1]):
            S_R[label] -= 1
            S_L[label] += 1

            if xj_sorted[idx + 1] == xj_sorted[idx]:
                continue

            gain = self._information_gain(S_all, S_L, S_R)

            if gain > best_gain:
                best_gain = gain
                best_split = (xj_sorted[idx] + xj_sorted[idx + 1]) * 0.5

        if best_split is not None and best_gain > gain_threshold:
            return Split(feature=feature, threshold=float(best_split), gain=float(best_gain))

        return None

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, gain_threshold: float = 0.0
    ) -> Split | None:
        _, d = X.shape

        best_split: Split | None = None
        for j in range(d):
            split = self._best_threshold_for_feature(X[:, j], y, j, gain_threshold)
            if split is None:
                continue
            if (best_split is None) or (split.gain > best_split.gain):
                best_split = split
        return best_split

    # ---------------------- Tree Builder (private) ----------------------

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n, _ = X.shape

        # Edge case
        if n == 0:
            return Node(label=0, depth=depth)

        # Leaf node : All labels are the same
        if np.all(y == y[0]):
            return Node(label=int(y[0]), depth=depth)

        # Check hyperparameters
        if depth >= self.config.max_depth or n < self.config.min_samples_split:
            mode = int(np.argmax(np.bincount(y, minlength=self._n_classes)))
            return Node(label=mode, depth=depth)

        # Find best split
        split = self._best_split(X, y, gain_threshold=self.config.gain_threshold)
        if split is None:
            mode = int(np.argmax(np.bincount(y, minlength=self._n_classes)))
            return Node(label=mode, depth=depth)

        # Split and recurse
        j, t = split.feature, split.threshold
        mask_left = X[:, j] < t
        left_child = self._build_tree(X[mask_left], y[mask_left], depth + 1)
        right_child = self._build_tree(X[~mask_left], y[~mask_left], depth + 1)

        return Node(feature=j, threshold=t, left=left_child, right=right_child, depth=depth)

    # ---------------------- Pruning  ----------------------

    def prune(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> None:
        """Post-pruning"""
        if self.root is None:
            raise ValueError("The tree has not been trained yet")

        # Make Each node store which training and validation samples reach them.
        def _attach_data_indices(node: Node, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
            node.train_idx = train_idx
            node.val_idx = val_idx

            if node.label is not None:
                return

            train_mask = X_train[train_idx, node.feature] < node.threshold
            val_mask = X_val[val_idx, node.feature] < node.threshold

            left_train_idx = train_idx[train_mask]
            right_train_idx = train_idx[~train_mask]
            left_val_idx = val_idx[val_mask]
            right_val_idx = val_idx[~val_mask]

            _attach_data_indices(node.left, left_train_idx, left_val_idx)
            _attach_data_indices(node.right, right_train_idx, right_val_idx)

        def _valdiation_error(node: Node) -> float:
            # If leaf node
            if node.label is not None:
                if (n_val := len(node.val_idx)) == 0:
                    return 0.0
                predictions = np.full(n_val, node.label, dtype=np.int64)
            else:
                predictions = self._predict_subtree(node, X_val[node.val_idx])
            return np.mean(predictions != y_val[node.val_idx]) if len(node.val_idx) != 0 else 0.0

        def _mode_label(node: Node) -> int:
            labels = y_train[node.train_idx]
            if len(labels) == 0:
                return 0
            return int(np.argmax(np.bincount(labels, minlength=self._n_classes)))

        def _prune_node(node: Node) -> bool:
            """Recursively prune a node. Return True if pruned anywhere lower in the subtree"""
            if node.label is not None:
                return False

            # Recurse on children (depth-first)
            prune_left = _prune_node(node.left)
            prune_right = _prune_node(node.right)

            # Check if Pruneable
            if node.left.label is not None and node.right.label is not None:
                error = _valdiation_error(node)

                # Error if pruned
                majority_label = _mode_label(node)
                temp_leaf = Node(label=majority_label)

                temp_leaf.train_idx = node.train_idx
                temp_leaf.val_idx = node.val_idx

                error_after_pruning = _valdiation_error(temp_leaf)

                prune_gain = (
                    error - error_after_pruning - self.config.alpha
                )  # NOTE: alpha is set to 0 in the assignment
                if prune_gain >= -self.config.tol:  # Note: tol is set to 0 in the assignment
                    node.left = None
                    node.right = None
                    node.label = majority_label
                    return True

            return prune_left or prune_right

        # add data indices
        _attach_data_indices(self.root, np.arange(X_train.shape[0]), np.arange(X_val.shape[0]))

        # Possibly add a max possible iterations on a pruning
        # Set true so starts with at least 1 pass
        while _prune_node(self.root):
            continue

    def _predict_subtree(self, node: Node, X: np.ndarray) -> np.ndarray:
        """Predict only withing a subtree"""
        if node.label is not None:
            return np.full(X.shape[0], node.label, dtype=np.int64)
        return np.array([node.predict(x) for x in X], dtype=np.int64)
