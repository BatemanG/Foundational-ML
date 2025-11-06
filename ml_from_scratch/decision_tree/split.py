from dataclasses import dataclass

@dataclass(frozen=True)
class Split:
    feature: None | int
    threshold: float
    gain: float
