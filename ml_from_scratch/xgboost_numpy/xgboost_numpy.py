"""
xgboost_numpy.py

A from scratch impoermtation of the Gradient Boosting Tree from binary classification

Reference:
        [1] T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System. 2016.
        [2] G. Ke et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Protocol
import numpy as np
from collections import defaultdict

from ._base import BoosterConfig, SplitConfig, SplitInfo, Node, Loss
from .loss import MultiClassLogLoss

# ===================================================================================
#  Exclusive Feature Bundliny (EFB)
# ===================================================================================

class ExclusiveFeatureBundler:
    """
    Implements of EFB by modeling features as a graph and suing greedy colouring
    to find bundles of mutually exclusive features.
    """
    def __init__(self, max_bins: int = 255, collision_threshold: float = 0.1):
        self.max_bins = max_bins
        self.collision_threshold = collision_threshold
        self.bundles: Optional[List[List[int]]] = None
        self.bin_offsets: Optional[np.ndarray] = None
        self.bin_mappers: Optional[List[np.ndarray]] = None
        self.n_bundled_features: Optional[int] = None

    def _build_feature_graph(self, X: np.ndarray) ->List[List[int]]:
        """Builds a graph representation of the features"""
        n_samples, n_features = X.shape
        non_zero_indices = [np.where(X[:, i] != 0)[0] for i in range(n_features)]

        adjaceny_matrix = defaultdict(list)
        for i in range(n_features):
            for j in range(i+1, n_features):
                if len(non_zero_indices[i]) == 0 or len(non_zero_indices[j]) == 0:
                    continue
                n_common_indices = len(np.intersect1d(non_zero_indices[i], non_zero_indices[j]))
                if n_common_indices / n_samples > self.collision_threshold:
                    adjaceny_matrix[i].append(j)
                    adjaceny_matrix[j].append(i)

        return [sorted(neighbors) for i, neighbors in sorted(adjaceny_matrix.items())]

    def _greedy_colouring(self, adjaceny_matrix: List[List[int]], n_features: int) -> List[int]:
        """Assigns a bundle index (colour) to each feature greddily."""
        colours = np.full(n_features, -1)
        for i in range(n_features):
            neighbor_colours = {colours[j] for j in adjaceny_matrix[i] if colours[j] != -1}
            colour = 0
            while colour in neighbor_colours:
                colour += 1
            colours[i] = colour
        return colours

    def fit(self, X: np.ndarray) -> None:
        """Builds a graph representation of the features and assigns colours to them"""
        n_samples, n_features = X.shape
        adjaceny_matrix = self._build_feature_graph(X)
        colours = self._greedy_colouring(adjaceny_matrix, n_features)


        self.n_bundled_features = max(colours) + 1
        self.bundles = [[] for _ in range(self.n_bundled_features)]
        for feature_idx, bundle_idx in enumerate(colours):
            self.bundles[bundle_idx].append(feature_idx)

        # Create bin mappers and offsets for the transform step
        self.bin_mappers = []
        for feature_idx in range(n_features):
             # Simple quantile binning for demonstration
            quantiles = np.unique(np.quantile(X[:, feature_idx], np.linspace(0, 1, self.max_bins + 1)))
            self.bin_mappers.append(quantiles)

        self.bin_offsets = np.zeros(self.n_bundled_features, dtype=int)
        for i, bundle in enumerate(self.bundles):
            if i > 0:
                self.bin_offsets[i] = self.bin_offsets[i-1]
                for f_idx in self.bundles[i-1]:
                    self.bin_offsets[i] += len(self.bin_mappers[f_idx]) - 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms X into a dataset with bundled features."""
        if self.bundles is None:
            raise RuntimeError("Bundler has not been fitted.")

        X_binned = np.empty_like(X, dtype=int)
        for feature_idx in range(X.shape[1]):
            X_binned[:, feature_idx] = np.digitize(X[:, feature_idx], self.bin_mappers[feature_idx][1:], right=True)

        X_bundled = np.zeros((X.shape[0], self.n_bundled_features))
        for bundle_idx, bundle in enumerate(self.bundles):
            offset = self.bin_offsets[bundle_idx]
            current_offset = 0
            for feature_idx in bundle:
                X_bundled[:, bundle_idx] += X_binned[:, feature_idx] + offset + current_offset
                current_offset += len(self.bin_mappers[feature_idx]) - 1
        return X_bundled

# ===================================================================================
#  Tree Builder (with GOSS)
# ===================================================================================
class TreeBuilder:
    """Builds a single decision tree for a given set of Gradients and Hessians."""
    def __init__(self, X: np.ndarray, g: np.ndarray, h:np.ndarray, split_config: SplitConfig):
        self.X = X; self.g = g; self.h = h
        self.split_config = split_config
        self.n_samples, self.n_features = X.shape

    def build(self) -> Node:
        root_indices = np.arange(self.n_samples)
        return self._build_recursive(root_indices, depth=0)

    def _calculate_leaf_weight(self, indices: np.ndarray) -> float:
        """Calculates optimal leaf weight w_{j}^{*} = - \frac{G_{j}}{ H_{j}+ \lambda}"""
        g_sum, h_sum = self.g[indices].sum(), self.h[indices].sum()
        return - g_sum / (h_sum + self.split_config.lambda_)

    def _calculate_gain(self, g_parent: float, h_parent: float, g_left: float, h_left: float) -> float:
        """ Compute split gain
        Gain &=  Loss_{Parent} - (Loss_{Left} + Loss_{Right}) \\
        &= \frac{1}{2} \left[ \frac{G_{L}^{2}}{H_{L} + \lambda} + \frac{G_{R}^{2}}{ H_{R} + \gamma}
        - \frac{G_{P}^{2}}{H_{P} + \gamma}\right] - \gamma
        \end{align*}
        """
        g_right = g_parent - g_left
        h_right = h_parent - h_left

        if h_left < self.split_config.min_hessian or h_right < self.split_config.min_hessian:
            return - 1.0 # Dont split

        score_left = (g_left ** 2) / (h_left + self.split_config.lambda_)
        score_right = (g_right ** 2) / (h_right + self.split_config.lambda_)
        score_parent = (g_parent ** 2) / (h_parent + self.split_config.lambda_)
        return 0.5 * (score_left + score_right - score_parent) - self.split_config.gamma

    def _find_best_split(self, indices:np.ndarray) -> Optional[SplitInfo]:
        """Finds the best split for a node using a Greedy algorithm.
        Complexity O( d * n log n ), d = n_features, n=len(indices)."""

        if len(indices) < 2:
            return None

        max_gain = 0.0
        best_split: Optional[SplitInfo] = None
        g_parent, h_parent = self.g[indices].sum(), self.h[indices].sum()
        X_subset, g_subset, h_subset = self.X[indices], self.g[indices], self.h[indices]

        for feature_idx in range(self.n_features):
            feature_values = X_subset[:, feature_idx]
            sorted_indices = np.argsort(feature_values)
            x_sorted, g_sorted, h_sorted = feature_values[sorted_indices], g_subset[sorted_indices], h_subset[sorted_indices]

            g_left = h_left = 0.0
            for i in range(len(indices) - 1):
                # Moving from right to left node
                g_left += g_sorted[i]
                h_left += h_sorted[i]
                if x_sorted[i] == x_sorted[i+1]:
                    continue #Optimal point will always be between new information

                current_gain = self._calculate_gain(g_parent, h_parent, g_left, h_left)
                if current_gain > max_gain:
                    max_gain = current_gain
                    threshold = 0.5 * ( x_sorted[i] + x_sorted[i+1] )
                    best_split = SplitInfo(
                        feature_idx=feature_idx, threshold=threshold, gain=max_gain,
                        g_left=g_left, h_left=h_left,
                        g_right=g_parent-g_left, h_right=h_parent-h_left
                    )
        return best_split

    def _get_goss_indice(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implements GOSS sampling logic.
        1. Sort instances by absolute gradient.
        2. Keep top `a`% (large gradient instances).
        3. Sample `b`% from the rest (small gradient instances).
        4. Return the combined indices and the weight for small gradient instances.
        """
        abs_g = np.abs(self.g[indices])
        sorted_g_indices = np.argsort(-abs_g)

        n_A = int(len(indices) * self.split_config.goss_a)
        n_A_complement_sample = int((len(indices) - n_A)  * self.split_config.goss_b)

        A = sorted_g_indices[:n_A]
        A_complement = sorted_g_indices[n_A:]
        A_complement_sample = np.random.choice(A_complement, size=n_A_complement_sample, replace=False)

        # Weight which is applied ot the small gradient instance to maintian an unbiased estimation
        weight = (1.0 - self.split_config.goss_a) / self.split_config.goss_b
        return np.concatenate([A, A_complement_sample]), n_A, weight

    def _find_best_split_goss(self, indices: np.ndarray) -> Optional[SplitInfo]:
        """Finds the best split for a node using GOSS sampling.
        Complexity O( d * n log n ), d = n_features, n=len(indices)."""
        goss_indices, n_A, weight = self._get_goss_indice(indices)

        original_g = self.g.copy()
        original_h = self.h.copy()

        # Scale the small Gradient
        self.g[indices[goss_indices[n_A:]]] *= weight
        self.h[indices[goss_indices[n_A:]]] *= weight

        best_split = self._find_best_split(indices[goss_indices])

        #Restore Original
        self.g = original_g
        self.h = original_h

        return best_split

    def _build_recursive(self, indices: np.ndarray, depth: int) -> Node:
        """Builds a single decision tree for a given set of Gradients and Hessians."""
        node = Node(depth=depth, instance_indices=indices)

        if depth >= self.split_config.max_depth or len(indices) == 0 or self.h[indices].sum() < self.split_config.min_hessian:
            node.weight = self._calculate_leaf_weight(indices)
            return node

        if self.split_config.algorithm == "goss":
            split_info = self._find_best_split_goss(indices)
        else:
            split_info = self._find_best_split(indices)


        # Leaf case
        if split_info is None or split_info.gain < 0.0:
            node.weight = self._calculate_leaf_weight(indices)
            return node

        node.split_info = split_info
        feature_values = self.X[indices, split_info.feature_idx]
        left_mask = feature_values <= split_info.threshold
        right_mask = ~left_mask
        node.left_child = self._build_recursive(indices[left_mask], depth + 1)
        node.right_child = self._build_recursive(indices[right_mask], depth + 1)
        return node

# ===================================================================================
#  Booster Model
# ===================================================================================

class GBT:
    """Implements Gradient Boosting Tree for binary classification"""
    def __init__(self,
                 booster_config: BoosterConfig,
                 split_config: SplitConfig,
                 loss: Loss = MultiClassLogLoss(),
                 efb_bundler: Optional[ExclusiveFeatureBundler] = None,):
        self.booster_config = booster_config
        self.split_config = split_config
        self.loss = loss
        self.trees: List[List[Node]] = []
        self.initial_prediction: Optional[np.ndarray] = None
        self.n_classes: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the multi-class gradient boosting model
        Builds n_classes trees for each n_estimators (iterations)
        """
        self.n_classes = len(np.unique(y))
        y_one_hot = np.eye(self.n_classes)[y]
        n_samples = X.shape[0]

        # Initial predictions: log-odds for each class based on freq
        class_count = np.bincount(y)
        class_propertions = class_count / n_samples
        self.initial_prediction = np.log( class_propertions/ (1 - class_propertions) )
        y_pred_loigt = np.full((n_samples, self.n_classes), self.initial_prediction)

        for _ in range(self.booster_config.n_estimators):
            g, h = self.loss.compute_gh(y_one_hot, y_pred_loigt)

            current_estimator_trees: List[Node] = []
            for k in range(self.n_classes):
                # For each class, build a tree on its specifig gradients and gessns.
                builder = TreeBuilder(X, g[:, k], h[:, k], self.split_config)
                tree = builder.build()
                current_estimator_trees.append(tree)

                # Update the logit predictions for the current class.
                update = self._predict_single_tree(tree, X)
                y_pred_loigt[:, k] += self.booster_config.learning_rate * update

            self.trees.append(current_estimator_trees)

    def predict_probabilitys(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the probability of each class for a given set of data
        (n_samples, n_classes)
        """
        if self.initial_prediction is None:
            raise ValueError("Model has not been fitted yet.")

        n_samples = X.shape[0]
        y_pred_logit = np.full((n_samples, self.n_classes), self.initial_prediction)

        # Sum predicionts from all trees for all classes.
        for i in range(self.booster_config.n_estimators):
            for k in range(self.n_classes):
                tree = self.trees[i][k]
                y_pred_logit[:, k] += self.booster_config.learning_rate * self._predict_single_tree(tree, X)

        # Softmax to get probaiblitys from logits
        return self._softmax(y_pred_logit)


    def _predict_single_tree(self, tree: Node, X: np.ndarray) -> np.ndarray:
        """Predicts the gradient for a single tree on a given set of data"""
        n_samples = X.shape[0]
        predictions = np.empty(n_samples)

        for i in range(n_samples):
            node = tree
            while not node.is_leaf:
                if X[i, node.split_info.feature_idx] <= node.split_info.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
            predictions[i] = node.weight
        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_probabilitys(X)
        if self.n_classes == 2:
            return (probs[:, 1] > 0.5).astype(int)
        return np.argmax(probs, axis=1)

    @staticmethod
    def _softmax(y_pred_logit: np.ndarray) -> np.ndarray:
        """Vectorized Stable softmax calculation"""
        exp_logits = np.exp(y_pred_logit - np.max(y_pred_logit, axis=1, keepdims=True))  #Note: we take away the max to keep numbers small
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True) # (N,K)
