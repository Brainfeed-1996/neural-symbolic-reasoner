import pytest

from src.reasoner import InferenceService
from src.rule_engine import Rule, validate_truth_assignments


def test_forward_chaining_derives_consequent():
    svc = InferenceService(
        [
            Rule(("rain",), "wet_ground", 0.9),
            Rule(("wet_ground",), "slippery", 0.8),
        ]
    )

    result = svc.run({"rain": 1.0})

    assert result["wet_ground"] == 0.9
    assert result["slippery"] == pytest.approx(0.72)


def test_validate_truth_assignments_rejects_invalid_values():
    try:
        validate_truth_assignments({"bad": 1.1})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "must be in [0, 1]" in str(exc)
