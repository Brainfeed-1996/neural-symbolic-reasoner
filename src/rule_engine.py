from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

TruthAssignments = Dict[str, float]


@dataclass(frozen=True)
class Rule:
    """Simple Horn-style rule: if all antecedents are true, imply consequent."""

    antecedents: Tuple[str, ...]
    consequent: str
    weight: float = 1.0


class SymbolicRuleEngine:
    """Forward-chaining fuzzy rule engine used by the neural wrapper."""

    def __init__(self, *, threshold: float = 1e-6) -> None:
        self.threshold = threshold

    def infer(self, beliefs: TruthAssignments, rules: Iterable[Rule]) -> TruthAssignments:
        state = dict(beliefs)
        changed = True
        while changed:
            changed = False
            for rule in rules:
                antecedent_truth = min(state.get(a, 0.0) for a in rule.antecedents)
                inferred = antecedent_truth * rule.weight
                if inferred - state.get(rule.consequent, 0.0) > self.threshold:
                    state[rule.consequent] = inferred
                    changed = True
        return state


def validate_truth_assignments(values: TruthAssignments) -> None:
    for key, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Truth value for '{key}' must be in [0, 1], got {value}")
