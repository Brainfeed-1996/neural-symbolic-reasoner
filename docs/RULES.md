# Rules Guide

This document describes how to create and use rules in the Neural-Symbolic Reasoner.

## Rule Types

### 1. Constraint (Hard Rule)

Constraints are **must-satisfy** rules. If violated, the prediction is rejected.

```python
from neural_symbolic_reasoner.rules import Constraint

constraint = Constraint(
    name="age_valid",
    description="Age must be between 0 and 120",
    condition=lambda x: 0 <= x.get("age", 0) <= 120,
    action="reject"  # or "flag", "adjust"
)
```

### 2. Rule (Soft Override)

Soft rules adjust the prediction score based on conditions.

```python
from neural_symbolic_reasoner.rules import Rule

rule = Rule(
    name="vip_boost",
    description="VIP customers get a 20% score boost",
    condition=lambda x: x.get("is_vip", False),
    adjustment=0.2,
    operation="multiply"  # or "add"
)
```

## Creating Custom Rules

### Basic Template

```python
from neural_symbolic_reasoner.rules.base import BaseRule

class MyRule(BaseRule):
    def __init__(self, name, param1, param2):
        super().__init__(name)
        self.param1 = param1
        self.param2 = param2
    
    def evaluate(self, context):
        """Evaluate the rule against input context."""
        return self.param1 <= context.get("value", 0) <= self.param2
    
    def apply(self, score, context):
        """Modify the score based on evaluation."""
        if self.evaluate(context):
            return score + 0.1
        return score
```

### Advanced: Composed Rules

```python
class ComposedRule(BaseRule):
    def __init__(self, rules):
        self.rules = rules
    
    def evaluate(self, context):
        return all(rule.evaluate(context) for rule in self.rules)
```

## Rule Evaluation Order

Rules are evaluated in the order they're added:

1. **Constraints** are checked first
2. If all constraints pass, **rules** are applied
3. Final score is computed by combining model output with rule adjustments

## Best Practices

### 1. Keep Rules Simple

Each rule should do one thing well.

```python
# Good: Single purpose
Constraint("age_valid", lambda x: 0 <= x["age"] <= 120)

# Avoid: Complex multi-part rules
Constraint("complex", lambda x: 
    0 <= x["age"] <= 120 and 
    x["income"] > 0 and 
    x["score"] > 0.5)
```

### 2. Use Descriptive Names

```python
# Good
Constraint("no_negative_income")

# Bad
Constraint("rule1")
```

### 3. Add Descriptions

```python
Rule(
    name="fraud_detected",
    description="Triggered when multiple risk factors present",
    condition=lambda x: x["risk_score"] > 0.8,
    adjustment=-0.5
)
```

### 4. Test Rules Independently

```python
# Test constraint
assert constraint.evaluate({"age": 25}) == True
assert constraint.evaluate({"age": -5}) == False

# Test rule
assert rule.evaluate({"is_vip": True}) == True
assert rule.evaluate({"is_vip": False}) == False
```

## Common Patterns

### Pattern 1: Threshold Rules

```python
Rule(
    name="high_value_customer",
    condition=lambda x: x["lifetime_value"] > 10000,
    adjustment=0.15,
    operation="add"
)
```

### Pattern 2: Exclusion Rules

```python
Constraint(
    name="country_excluded",
    condition=lambda x: x["country"] not in BLOCKED_COUNTRIES,
    action="reject"
)
```

### Pattern 3: Conditional Boost

```python
def create_boost_rule(name, condition, boost_amount):
    return Rule(
        name=name,
        condition=condition,
        adjustment=boost_amount,
        operation="multiply"
    )

# Usage
premium_boost = create_boost_rule(
    "premium_boost",
    lambda x: x["tier"] == "premium",
    1.25
)
```

## Debugging Rules

### Enable Debug Mode

```python
reasoner = NeuralSymbolicReasoner(
    model=model,
    rules=[rule1, rule2],
    debug=True
)

result = reasoner.predict(data)
# Prints detailed evaluation trace
```

### Inspect Rule Application

```python
result = reasoner.predict(data)
print(result.rule_audit)  # Shows which rules applied
```
