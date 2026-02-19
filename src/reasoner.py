from __future__ import annotations

from typing import Iterable, List

nn = object

from .rule_engine import Rule, SymbolicRuleEngine, validate_truth_assignments


class SymbolicReasoner:
    """Hybrid neural-symbolic reasoner with lazy torch import."""

    def __init__(self, input_dim: int = 128, predicate_dim: int = 64):
        import torch.nn as torch_nn

        self.logic_layer = torch_nn.Linear(input_dim, predicate_dim)

    def forward(self, x):
        return self.logic_layer(x)


class InferenceService:
    """Framework-agnostic inference service for unit-testable symbolic reasoning."""

    def __init__(self, rules: Iterable[Rule]):
        self.rules: List[Rule] = list(rules)
        self.engine = SymbolicRuleEngine()

    def run(self, seed_beliefs: dict[str, float]) -> dict[str, float]:
        validate_truth_assignments(seed_beliefs)
        result = self.engine.infer(seed_beliefs, self.rules)
        validate_truth_assignments(result)
        return result


if __name__ == "__main__":
    print("Neural Symbolic Reasoner Initialized")
