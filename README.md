# Neural-Symbolic Reasoner

Hybrid reasoning project combining trainable neural representations with deterministic symbolic inference.

## Complexity tier

**Tier 3**: modular symbolic engine + optional neural projection + CI quality gates.

## Architecture

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
