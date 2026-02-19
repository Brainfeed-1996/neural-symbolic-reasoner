# Architecture

## Inference flow

1. Seed beliefs enter `InferenceService.run`.
2. Beliefs are validated to ensure fuzzy truth in `[0,1]`.
3. `SymbolicRuleEngine.infer` performs forward chaining until no rule can increase confidence.
4. Final beliefs are re-validated and returned.

## Design goals

- **Deterministic symbolic core** for reproducible reasoning.
- **Neural compatibility**: torch module kept optional and isolated.
- **Testability**: symbolic engine does not require GPU/torch runtime.
- **Extensibility**: add custom t-norm operators or rule schedulers without touching service interface.

## Quality gates

GitHub Actions workflow runs:
- Ruff lint
- Pytest suite
