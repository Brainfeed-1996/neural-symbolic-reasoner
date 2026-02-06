# Author: Olivier Robert-Duboille
# Description: Example demonstrating basic traffic light logic inference.

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.reasoner import SymbolicReasoner

def rule_traffic_light(logic, state):
    # If light is red -> stop
    # If light is green -> go
    
    red = state.get('is_red', torch.tensor(0.0))
    green = state.get('is_green', torch.tensor(0.0))
    
    should_stop = logic.implies_op(red, torch.tensor(1.0)) # Strong implication
    # Simplified: actually rule is: red -> stop.
    # truth(stop) >= truth(red) * weight
    
    # Direct mapping for demo:
    # Inference: stop = red
    # Inference: go = green
    
    return {
        'action_stop': red,
        'action_go': green
    }

def main():
    reasoner = SymbolicReasoner(logic_type='lukasiewicz')
    reasoner.add_rule(rule_traffic_light)
    
    # Scene: Light is mostly red
    facts = {
        'is_red': torch.tensor(0.8),
        'is_green': torch.tensor(0.1)
    }
    
    print("Initial Facts:", facts)
    
    result = reasoner.infer(facts)
    
    print("\nInferred State:")
    for k, v in result.items():
        print(f"{k}: {v.item():.4f}")

if __name__ == "__main__":
    main()
