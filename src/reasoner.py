# Author: Olivier Robert-Duboille
# Description: Symbolic Reasoner for forward-chaining inference on neural predicates.

import torch
from .logic_layer import DifferentiableLogicLayer

class SymbolicReasoner:
    def __init__(self, logic_type='product'):
        self.logic = DifferentiableLogicLayer(logic_type)
        self.knowledge_base = []

    def add_rule(self, rule_fn):
        """
        Adds a python function representing a logic rule.
        The function should accept the logic layer and a state dict as arguments.
        """
        self.knowledge_base.append(rule_fn)

    def infer(self, facts):
        """
        Performs forward chaining reasoning based on facts.
        
        Args:
            facts (dict): Mapping of concept names to probability tensors.
            
        Returns:
            dict: Augmented facts including inferred concepts.
        """
        current_state = facts.copy()
        
        # Iterative fixed-point computation (simplified)
        for _ in range(3): # Fixed steps for demo
            new_inferences = {}
            for rule in self.knowledge_base:
                inferred = rule(self.logic, current_state)
                if inferred:
                    for k, v in inferred.items():
                        if k in current_state:
                            # Aggregate evidence (using OR)
                            current_state[k] = self.logic.or_op(current_state[k], v)
                        else:
                            current_state[k] = v
                            
        return current_state
