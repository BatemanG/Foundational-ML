"""This is the main script for my application."""

import argparse
from evaluate_datasets import evaluate_tree_dataset


def main() -> None:
    """This is the main entry point for the coursework runner."""
    parser = argparse.ArgumentParser(description="Run Decision Tree cross-validation experiments.")

    parser.add_argument(
        "--dataset",
        type=str,
        default="clean",
        choices=["clean", "noisy"],
        help="Dataset to use: 'clean' or 'noisy' (default: clean)",
    )
    parser.add_argument(
        "--prune",
        action="store_true",  # this is a boolean flag. if present, it's true.
        help="Apply pruning to the decision tree.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        required=True,  # this is a required param
        help="Maximum depth of the tree.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        required=True,
        help="Number of folds for cross-validation.",
    )

    # optional hyperparameters
    parser.add_argument(
        "--gain-threshold",
        type=float,
        default=0.0,
        help="Minimum information gain to split a node (default: 0.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Alpha value for pruning (if --prune is used) (default: 0.0)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="Tolerance for pruning (if --prune is used) (default: 0.0)",
    )

    args = parser.parse_args()

    print("=== Decision Tree Evaluation ===")
    print("Running with parameters:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Pruning: {args.prune}")
    print(f"  Max Depth: {args.max_depth}")
    print(f"  N-Folds: {args.n_folds}")
    print(f"  Gain Threshold: {args.gain_threshold}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Tolerance: {args.tol}")
    print("---------------------------------")

    evaluate_tree_dataset(
        dataset=args.dataset,
        prune_tree=args.prune,
        max_depth=args.max_depth,
        n_folds=args.n_folds,
        gain_threshold=args.gain_threshold,
        alpha=args.alpha,
        tol=args.tol,
    )


if __name__ == "__main__":
    main()
