# Foundational-ML: Machine Learning from First Principles

This repository contains a curated library of core machine learning algorithms implemented from scratch in pure, vectorized NumPy. This project serves as a rigorous, axiomatic study of ML, emphasizing the direct line from mathematical derivation to efficient, testable code.

The goal is not to replace libraries like `scikit-learn`, but to build a production-quality library that proves a fundamental understanding of *how* these algorithms work. Each model is built on a "Trilogy of Understanding":

1.  **Mathematics:** The complete mathematical derivations are included in the `research_notes/` directory.
2.  **Implementation:** The code is structured as an installable Python package (`ml_from_scratch/`) with a focus on clean, efficient, and vectorized operations.
3.  **Validation:** The implementations are validated against standard libraries (e.g., `sklearn`) in `notebooks/` and verified for correctness with unit tests in `tests/`.

---

## Project Structure
Foundational-ML/
│
├── .gitignore
├── README.md
├── pyproject.toml
│
├── data/
│   └── wifi_db/
│       ├── clean_dataset.txt
│       └── noisy_dataset.txt
│
├── ml_from_scratch/
│   │
│   ├── __init__.py
│   │
│   ├── decision_tree/
│   │   ├── __init__.py
│   │   ├── tree.py
│   │   ├── node.py
│   │   ├── split.py
│   │   └── config.py
│   │
│   ├── gradient_boosting/
│   │   ├── __init__.py
│   │   ├── xgboost.py
│   │   ├── _base.py
│   │   └── loss.py
│   │
│   └── shared/
│       ├── __init__.py
│       ├── metrics.py
│       ├── cross_validation.py
│       └── plotting.py
│
├── notebooks/
│   │
│   ├── 01_decision_tree_analysis.ipynb
│   └── 02_xgboost_benchmark.ipynb
│
├── research_notes/
│   │
│   ├── decision_tree.md
│   └── gradient_boosting.md
│
├── scripts/
│   │
│   └── run_tree_evaluation.py
│
└── tests/
    │
    ├── test_decision_tree/
    │   ├── test_tree.py
    │   ├── test_pruning.py
    │   └── test_cross_val.py
    │
    └── test_gradient_boosting/
        └── test_xgboost.py

---

## Models & Implementations

### 1. Tree-Based Models

#### Decision Tree (CART)

* **Math:** `research_notes/decision_tree.md`
    * Derivation of Gini impurity and Information Gain for split-finding.
    * Algorithm for recursive tree-building.
* **Implementation:** `ml_from_scratch/decision_tree/`
    * Includes `DecisionTreeClassifier` built from `Node` and `Split` logic.
    * Supports `max_depth`, `min_samples_split`, and `gain_threshold`.
.

#### Gradient Boosted Trees (XGBoost-Style)

* **Math:** `research_notes/gradient_boosting.md`
    * Derivation of the GBT algorithm as gradient descent in function space.
    * Full derivation of the objective function, gradient (residuals), and Hessian used for split-finding (Mean Squared Error loss).
* **Implementation:** `ml_from_scratch/gradient_boosting/`
    * Includes `XGBoostRegressor` built on a `BaseTree` learner.
    * Implemented in vectorized NumPy for efficient computation of gradients and optimal leaf values.