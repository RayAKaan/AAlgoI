from aalgoi.synthesis.validator import SynthesisValidator
from aalgoi.synthesis.promotion import PromotionManager


def test_validator_rejects_unsafe_import():
    validator = SynthesisValidator()
    result = validator.validate_code("import os")
    assert not result["safe"]


def test_validator_rejects_dangerous_call():
    validator = SynthesisValidator()
    result = validator.validate_code("eval('1+1')")
    assert not result["safe"]


def test_validator_accepts_safe_code():
    validator = SynthesisValidator()
    result = validator.validate_code("def solve(x): return x")
    assert result["safe"]


def test_validator_executes_with_data():
    validator = SynthesisValidator()
    result = validator.validate_code("def solve(x): return sorted(x)", data=[3, 1, 2])
    assert result["safe"]
    assert result["result"] == [1, 2, 3]


def test_promotion_basic():
    mgr = PromotionManager()
    result = mgr.promote("def solve(x): return x", name="identity")
    assert result["success"]


def test_promotion_rejects_unsafe():
    mgr = PromotionManager()
    result = mgr.promote("import sys")
    assert not result["success"]


def test_promotion_with_data_and_expected():
    mgr = PromotionManager()
    code = "def solve(x): return sorted(x)"
    result = mgr.promote(code, data=[3, 1, 2], expected=[1, 2, 3], name="sort")
    assert result["success"]


def test_promotion_detects_wrong_result():
    mgr = PromotionManager()
    code = "def solve(x): return x"
    result = mgr.promote(code, data=[3, 1, 2], expected=[1, 2, 3], name="sort")
    assert not result["success"]
    assert "expected" in result.get("error", "")
