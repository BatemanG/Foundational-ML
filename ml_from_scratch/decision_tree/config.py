from dataclasses import dataclass

@dataclass(frozen=True)
class DecisionTreeConfig:
    max_depth: int = 100
    min_samples_split: int = 2
    gain_threshold: float = 0.0
    alpha: float = 0.0
    tol: float = 0.0
