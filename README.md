# Neural-Symbolic Reasoner

Minimal neuro-symbolic patterns combining learned scoring with symbolic constraints.

## Features

- **Hybrid Inference**: Combine neural models with symbolic rule overrides
- **Rule-Based Constraints**: Add logical constraints to ML predictions
- **Interpretability**: Clear rule-based decision paths
- **Modular Design**: Easy to extend with new rules or models

## Usage

```python
from neural_symbolic_reasoner import NeuralSymbolicReasoner

# Initialize with rules and model
reasoner = NeuralSymbolicReasoner(
    model=model,
    rules=["rule1", "rule2"],
    weights=[0.7, 0.3]
)

# Get hybrid prediction
prediction = reasoner.predict(input_data)
```

## Notebooks

- `notebooks/01_neurosymbolic_rules_plus_linear_model.ipynb` — Hybrid inference with model + rule overrides

## Architecture

```
neural-symbolic-reasoner/
├── src/
│   ├── models/          # Neural models
│   ├── rules/           # Symbolic rules
│   └── reasoner.py      # Hybrid inference engine
├── notebooks/
├── examples/
└── tests/
```

## License

MIT
