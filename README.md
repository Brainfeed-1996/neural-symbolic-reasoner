# Neural-Symbolic Reasoner

Hybrid reasoning project combining trainable neural representations with deterministic symbolic inference.

## 📊 Live Dashboard
[🚀 Intelligence Monitoring](https://neural-symbolic-reasoner-dash.vercel.app)

## Features
- **Multi-Agent Reasoning**: Propagation nodes for distributed belief updates.
- **Deep Logic**: Forward-chaining engine with weighted fuzzy rules.
- **Elite Monitoring**: Real-time dashboard for inference tracking.

## Architecture
...

- `src/reasoner.py`
  - `SymbolicReasoner`: lazy torch neural projection layer.
  - `InferenceService`: framework-agnostic orchestration for symbolic inference.
- `src/rule_engine.py`
  - `Rule` dataclass for weighted Horn-style implications.
  - `SymbolicRuleEngine` with forward chaining convergence.

See [ARCHITECTURE.md](ARCHITECTURE.md).

## Validation

```bash
pip install -e .[dev]
ruff check src tests
pytest -q
```
