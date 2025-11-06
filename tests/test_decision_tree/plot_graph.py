"""Simple script for saving an image for plotting in the report folder."""

import numpy as np
import matplotlib.pyplot as plt

from decision_tree.plotting import plot_tree
from decision_tree.config import DecisionTreeConfig
from decision_tree.cross_validation import train_tree
from decision_tree.pairgrid import pairgrid_tree_regions


def load_wifi_txt(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load data."""
    arr = np.loadtxt(path)
    X = arr[:, :-1].astype(np.float64)
    y_raw = arr[:, -1].astype(np.int64)
    y = y_raw - 1  # Since we want to start with 0
    return X, y


X_clean, y_clean = load_wifi_txt("wifi_db/clean_dataset.txt")

max_depth = 50


config = DecisionTreeConfig(max_depth=max_depth)

tree_clean = train_tree(X_clean, y_clean, None, None, config=config, prune=False)

plot_tree(tree_clean.root, 4, figsize=(25, 10))
plt.title("")
plt.savefig("report/resources/pdf/clean_graph.pdf")
plt.close()

pairgrid_tree_regions(X_clean, y_clean, tree_clean, show_points=True, draw_thresholds=False)
plt.savefig("report/resources/pdf/clean_pairgrid.pdf")
print("Graphs succesfully plotted and saved.")
