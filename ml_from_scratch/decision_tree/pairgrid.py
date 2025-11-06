"""Construct pairgrid plot for the decision tree."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def collect_splits(node, depth=0, splits=None):
    if splits is None:
        splits = []
    # base case :Lead
    if node is None or node.label is not None:
        return splits
    splits.append((node.feature, node.threshold, depth))
    collect_splits(node.left, depth + 1, splits)
    collect_splits(node.right, depth + 1, splits)
    return splits


def pairgrid_tree_regions(
    X,
    y,
    tree,
    class_names=None,
    grid_points=100,
    cmap=None,
    figsize_scale=2.5,
    scatter_alpha=0.5,
    show_points=True,
    draw_thresholds=False,
    fig=None,
    axes=None,
    X_ref=None,
):
    n_samples, d = X.shape

    classes = np.unique(y)
    k = len(classes)
    if class_names is None:
        class_names = [str(c + 1) for c in classes]  # 1, 2, ... k

    # if cmap is None:
    tab10 = plt.get_cmap("tab10")  # 10 distinct strong colors
    colors = [tab10(i) for i in range(k)]
    cmap = ListedColormap(colors)

    if X_ref is None:
        X_ref = X
    mins, maxs, median = X_ref.min(axis=0), X_ref.max(axis=0), np.median(X_ref, axis=0)
    pads = 0.05 * (maxs - mins + 1e-10)
    mins -= pads
    maxs += pads

    if fig is None or axes is None:
        fig, axes = plt.subplots(d, d, figsize=(figsize_scale * d, figsize_scale * d))
    # add d =1 case

    # FOr Thresholds
    if draw_thresholds:
        splits = collect_splits(tree.root)
        max_depth = max(d for _, _, d in splits) if splits else 1

    for i in range(d):  # row
        for j in range(d):  # col
            ax = axes[i, j]

            if i == j:
                # diangonal as a 1d histogram
                ax.hist(
                    X[:, i], bins=30, density=True, color="royalblue", edgecolor="black", alpha=0.7
                )
                ax.set_xlim(mins[j], maxs[j])

                if i == 0:
                    ax.set_ylabel(f"$x_{j}$", fontsize=20)
                    ax.set_yticks([])
                    ax.yaxis.set_label_coords(-0.21, 0.5)
                    continue
                if i == d - 1:
                    ax.set_xlabel(f"$x_{j}$", fontsize=20)
                    ax.set_yticks([])
                    continue

                ax.set_yticks([])
                continue

            x1 = np.linspace(mins[j], maxs[j], grid_points)
            x2 = np.linspace(mins[i], maxs[i], grid_points)
            XX, YY = np.meshgrid(x1, x2)

            grid = np.tile(median, (grid_points * grid_points, 1))
            grid[:, j] = XX.ravel()
            grid[:, i] = YY.ravel()

            predictions = tree.predict(grid).reshape(grid_points, grid_points)

            ax.imshow(
                predictions,
                alpha=0.3,
                origin="lower",  # flip so y increases upward
                extent=(mins[j], maxs[j], mins[i], maxs[i]),
                aspect="auto",  # make all plots sqaure
                cmap=cmap,
            )

            if show_points:
                for c_idx, c in enumerate(classes):
                    mask = y == c
                    ax.scatter(
                        X[mask, j],
                        X[mask, i],
                        alpha=scatter_alpha,
                        edgecolors="white",
                        linewidths=0.8,
                        s=20,
                        color=colors[c_idx],
                        label=class_names[c_idx],
                    )

            if draw_thresholds:
                for feature, thr, depth in splits:
                    alpha = 0.5 * (1 - depth / (max_depth + 1))
                    lw = 1 - depth / (max_depth + 1)

                    # if the split feature is in current plot
                    if feature == j:
                        ax.axvline(x=thr, color="black", alpha=alpha, lw=lw)
                    elif feature == i:
                        ax.axhline(y=thr, color="black", alpha=alpha, lw=lw)

            if i < d - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(f"$x_{j}$", fontsize=20)

            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(f"$x_{i}$", fontsize=20)

            ax.set_xlim(mins[j], maxs[j])
            ax.set_ylim(mins[i], maxs[i])

    # fake point for legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",  # draw a circular marker
            lw=0,  # no line connecting points
            ms=10,  # marker size
            mec="white",  # marker edge color
            mew=0.6,  # marker edge width
            mfc=colors[c],  # marker fill color (same as scatter color)
        )
        for c in classes
    ]

    fig.legend(
        handles,
        class_names,  # labels to show beside each dot
        loc="upper center",  # where to place the legend
        ncol=min(k, 6),  # spread across up to 6 columns
        frameon=False,  # remove box outline
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes
