import numpy as np
import matplotlib.pyplot as plt
from .node import Node


def _plot_recursive(
    ax,
    node: Node,
    max_depth: int,
    x: float,
    y: float,
    scale: float,
    v_dist: float,
):
    """The core recursive plotting function (internal use)."""
    if node is None or node.depth > max_depth:
        return

    if node.label is not None:
        label = f"{node.label}"
        node_style = dict(
            ha="center", va="center", bbox=dict(boxstyle="round4,pad=0.5", fc="yellow")
        )

    else:
        label = f"$x_{node.feature}$ < {node.threshold:.2f}"
        node_style = dict(
            ha="center", va="center", bbox=dict(boxstyle="round4,pad=0.3", fc="lightblue")
        )

    ax.text(x, y, label, **node_style, fontsize=12)

    # Base case: if the node is a leaf, we don't need to draw its children
    if node.label is not None:
        return

    # Calculate child positions
    y_child = y - v_dist
    # child_scale = scale * 0.5
    # Compute how much space left/right subtrees need
    left_w = _compute_subtree_width(node.left)
    right_w = _compute_subtree_width(node.right)
    total_sub_w = max(1, left_w + right_w)

    # scale = scale / total_sub_w

    if node.left and node.left.depth <= max_depth:
        # x_left = x - child_scale
        x_left = x - (right_w/total_sub_w) * scale
        ax.plot([x, x_left], [y, y_child], "k-")
        _plot_recursive(ax, node.left, max_depth, x_left, y_child, scale * 0.5, v_dist)

    if node.right and node.right.depth <= max_depth:
        # x_right = x + child_scale
        x_right = x + (left_w/total_sub_w) * scale
        ax.plot([x, x_right], [y, y_child], "k-")
        _plot_recursive(ax, node.right, max_depth, x_right, y_child, scale * 0.5, v_dist)


def _compute_subtree_width(node: Node | None ) -> int:
    """Count total leaves under this node to allocate horizontal space"""
    if node is None:
        return 0
    if node.label is not None:
        return 1
    return _compute_subtree_width(node.left) + _compute_subtree_width(node.right)


def plot_tree(tree_root: Node, max_depth: int, figsize: tuple = (30, 9)) -> None:
    """Wrapper function to plot the decision node."""
    _, ax = plt.subplots(figsize=figsize)

    # h_spread = 2**max_depth
    total_w = _compute_subtree_width(tree_root)
    scale = total_w

    _plot_recursive(
        ax=ax, node=tree_root, max_depth=max_depth, x=0, y=0, scale=scale, v_dist=2.0
    )

    ax.axis("off")  # cleaner look
    plt.tight_layout()
    plt.title(f"Decision Tree (up to depth {max_depth})", fontsize=15, fontweight="bold")
