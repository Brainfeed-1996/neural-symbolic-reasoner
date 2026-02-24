from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

TruthAssignments = Dict[str, float]


@dataclass(frozen=True)
class Rule:
    """Simple Horn-style rule: if all antecedents are true, imply consequent."""

    antecedents: Tuple[str, ...]
    consequent: str
    weight: float = 1.0


class ReasoningNode:
    """A node in a multi-agent reasoning graph."""
    def __init__(self, name: str, engine: 'SymbolicRuleEngine'):
        self.name = name
        self.engine = engine
        self.peers: List['ReasoningNode'] = []
        self.local_beliefs: TruthAssignments = {}

    def connect(self, other: 'ReasoningNode'):
        self.peers.append(other)

    def update_beliefs(self, new_beliefs: TruthAssignments):
        self.local_beliefs.update(new_beliefs)
        self.propagate()

    def propagate(self):
        """Propagate inferred beliefs to peers."""
        # Simple propagation logic: share derived facts
        for peer in self.peers:
            peer.receive_beliefs(self.local_beliefs, source=self.name)

    def receive_beliefs(self, incoming: TruthAssignments, source: str):
        # Merge logic (e.g., max truth value)
        changed = False
        for k, v in incoming.items():
            if v > self.local_beliefs.get(k, 0.0):
                self.local_beliefs[k] = v
                changed = True
        if changed:
            print(f"Node {self.name} updated from {source}")


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
                if not rule.antecedents:
                    continue
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
