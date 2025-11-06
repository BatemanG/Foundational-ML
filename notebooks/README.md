# Decision-Trees

## Introduction 
This directory contains all the work done on the ``CW-70050-Decision-Trees`` group project. This document contains instructions to get the code up and running, as well as a more detailed overview of our codebase. All other relevant information is contained in our report, which has been submitted separately. 

##  Getting Started

Follow these instructions to set up the development environment.

## Prerequisites
Before you begin, ensure you have the following tools installed:

1. `git`
2. `pyenv` for managing Python versions.



## Commands

1.  **Run 10-fold cross-validation on a dataset:**
    ```bash
    python3 main.py --dataset path/to/data.txt --n-folds 10
    ```

2.  **Run 10-fold cross-validation with pruning:**
    ```bash
    python3 main.py --dataset path/to/data.txt --n-folds 10 --prune
    ```

3.  **Run 5-fold cross-validation with pruning:**
    ```bash
    python3 main.py --dataset ~/Documents/sample-copy-test-data.txt --max-depth 10 --n-folds 5
    ```

While the above three commands are probably sufficient to test the code on a dataset of your choice, we have included many extra hyperparamters, that may be of interest. A full table of commands is shown below.

Argument | Type | Default | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--dataset` | str | `"clean"` | No | Dataset to use: 'clean', 'noisy', or a custom file path. |
| `--n-folds` | int | - | **Yes** | Number of folds for cross-validation. |
| `--prune` | bool | `False` | No | Apply pruning to the decision tree. |
| `--max-depth` | int | `100` | No | Maximum depth of the tree. |
| `--gain-threshold` | float | `0.0` | No | Minimum information gain required to split a node. |
| `--alpha` | float | `0.0` | No | Alpha value for pruning (used if `--prune` is set). When used require larger gains for pruning. |
| `--tol` | float | `0.0` | No | An additinal pruning hyperparamter, allowing for pruning with small negative prune gains (used if ``--prune`` is set).

## Bonus: Pruning & Visualisation Demo

As part of the project requirements, we implemented a function to visualise the decision tree:

We have created a separate script, `experiments/demo_pruning.py`, which demonstrates our pruning logic and also executes the tree visualization.

To run this specific demo, use the following command:

```bash
python3 -m experiments.demo_pruning
```

The plot results are then shown in ``report/resources/pdf``.

## Codebase overview 

The project is organized into the following main directories and files:

| Directory/File | Description |
| :--- | :--- |
| `decision_tree/` | Core source code for the custom Decision Tree implementation. |
| `experiments/` | Scripts used to run various experiments (e.g., pruning demos, dataset evaluation). |
| `nbs/` | Jupyter Notebooks used for initial exploration. |
| `report/` | All files related to the final report. |
| `wifi_db/` | Contains the dataset files used for training and evaluation. |
| `main.py` | The main script to run the final version of the decision tree model or experiments. |
| `requirements.txt` | A list of all necessary Python dependencies to run the project. |
| `pyproject.toml` | Project configuration file. |

### Core Decision Tree Implementation (`decision_tree/`)

| File | Purpose |
| :--- | :--- |
| `tree.py` | Contains the main `DecisionTree` classifier class, as well as all training logic. |
| `node.py` | Defines the `Node` class used to build the tree structure. |
| `split.py` | Defines the split class for the optimal split point. |
| `cross_validation.py` | Functions for running K-fold cross-validation. |
| `metrics.py` | Functions for calculating performance metrics (accuracy, precision, recall). |
| `plotting.py` | Contains the main tree plotting functions. |
| `utils.py` | General helper functions. |
| `pairgrid.py` | Contains pairgrid plot plotting function. |
