# Neural-Symbolic Reasoner

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Hybrid AI](https://img.shields.io/badge/Hybrid-AI-blue?style=flat-square)
![GitHub Stars](https://img.shields.io/github/stars/neural-symbolic-reasoner?style=flat-square)

</div>

Minimal neuro-symbolic patterns combining learned scoring with symbolic constraints for interpretable AI decision-making.

## Overview

This project bridges **machine learning** and **symbolic reasoning** to produce:

- **Interpretable predictions** with clear decision paths
- **Rule-based overrides** for domain expertise
- **Hybrid inference** combining neural and symbolic methods

## Key Concepts

### Hybrid Inference

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Neural    │────▶│   Combine    │────▶│  Final Decision │
│   Model     │     │   & Override │     │  + Explanation │
└─────────────┘     └──────────────┘     └─────────────────┘
        │                    │
        ▼                    ▼
   [scores]          [symbolic rules]
```

### Rule Types

| Rule Type | Description | Example |
|-----------|-------------|---------|
| Hard Constraint | Must satisfy, reject if violated | "age >= 0" |
| Soft Override | Adjust score based on rule | "if VIP: score * 1.5" |
| Filter | Remove invalid predictions | "exclude if country = blocked" |

## Installation

```bash
pip install -e .
pip install -e ".[dev]"  # For development
```

## Quick Start

```python
from neural_symbolic_reasoner import NeuralSymbolicReasoner
from sklearn.linear_model import LogisticRegression

# Create a simple model
model = LogisticRegression()

# Initialize reasoner with rules and weights
reasoner = NeuralSymbolicReasoner(
    model=model,
    rules=["rule1", "rule2"],
    weights=[0.7, 0.3]
)

# Get hybrid prediction
prediction = reasoner.predict(input_data)
```

## Usage

### Defining Rules

```python
from neural_symbolic_reasoner.rules import Rule, Constraint

# Hard constraint
constraint = Constraint(
    name="age_valid",
    condition=lambda x: x["age"] >= 0,
    message="Age must be non-negative"
)

# Soft override
override = Rule(
    name="vip_discount",
    condition=lambda x: x["is_vip"],
    adjustment=0.2,
    direction="multiply"
)

# Add rules to reasoner
reasoner.add_rules([constraint, override])
```

### Hybrid Prediction

```python
# Single prediction
result = reasoner.predict({
    "age": 25,
    "income": 50000,
    "is_vip": True
})

print(result.prediction)    # Final prediction
print(result.score)         # Confidence score
print(result.explanation)   # Rule-based explanation

# Batch prediction
results = reasoner.predict_batch(dataframe)
```

## Features

### Core Capabilities

- **Hybrid Inference**: Combine neural models with symbolic rules
- **Rule-Based Constraints**: Add logical constraints to ML predictions
- **Interpretability**: Clear rule-based decision paths
- **Modular Design**: Easy to extend with new rules or models
- **Explainability**: Generate human-readable explanations

### Supported Models

- scikit-learn classifiers and regressors
- PyTorch models (via wrapper)
- TensorFlow/Keras models (via wrapper)
- Custom model classes

### Rule Engine

- Boolean constraints
- Score adjustments (additive/multiplicative)
- Conditional overrides
- Rule composition

## Architecture

```
neural-symbolic-reasoner/
├── src/
│   ├── models/
│   │   ├── base.py           # Abstract base model
│   │   ├── sklearn_wrapper.py
│   │   └── pytorch_wrapper.py
│   ├── rules/
│   │   ├── base.py           # Abstract rule classes
│   │   ├── constraint.py     # Hard constraints
│   │   ├── override.py       # Soft overrides
│   │   └── engine.py         # Rule evaluation
│   ├── reasoner.py           # Main hybrid reasoner
│   └── explainer.py          # Explanation generation
├── notebooks/
│   └── 01_neurosymbolic_rules_plus_linear_model.ipynb
├── examples/
│   ├── basic_usage.py
│   └── advanced_rules.py
├── tests/
├── docs/
│   ├── RULES.md
│   └── API.md
└── pyproject.toml
```

## Notebooks

- `notebooks/01_neurosymbolic_rules_plus_linear_model.ipynb` — Hybrid inference with model + rule overrides

## Examples

### Credit Scoring

```python
# Rules based on domain expertise
rules = [
    Constraint("no_negative_income", lambda x: x["income"] >= 0),
    Rule("high_income_boost", lambda x: x["income"] > 100000, 0.1, "add"),
    Rule("recent_default", lambda x: x["has_default"], -0.5, "multiply")
]

reasoner = NeuralSymbolicReasoner(model, rules)
```

### Medical Diagnosis

```python
# Medical constraints
rules = [
    Constraint("age_range", lambda x: 0 <= x["age"] <= 120),
    Rule("symptom_weight", lambda x: x["fever"] > 38, 0.15, "add"),
    Constraint("required_symptoms", lambda x: x["symptom_count"] >= 2)
]
```

## Documentation

- [Rules Guide](docs/RULES.md) — Creating and managing rules
- [API Reference](docs/API.md) — Full API documentation
- [Examples](examples/) — Usage examples

## License

MIT
