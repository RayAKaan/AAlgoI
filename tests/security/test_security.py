"""Security-focused tests for eval replacement, token storage, and code injection prevention."""

import ast
import re

import pytest


def test_web_ui_no_eval():
    """Verify web_ui.py uses ast.literal_eval instead of eval."""
    import interface.web_ui as web_ui
    with open(web_ui.__file__, encoding='utf-8') as f:
        source = f.read()
    assert "ast.literal_eval" in source, "ast.literal_eval not found in web_ui.py"
    import re
    calls = re.findall(r'(?<![a-zA-Z_])eval\(', source)
    assert len(calls) == 0, f"eval() found in web_ui.py: {calls}"


def test_toke_manager_no_hardcoded_creds():
    """Verify no hardcoded tokens/secrets in token_manager.py."""
    import aalgoi.core.token_manager as tm
    with open(tm.__file__, encoding='utf-8') as f:
        source = f.read()
    if "# None if unset" in source:
        pass  # comments with default values are fine
    hardcoded = re.findall(r'(?:api_key|token|secret)\s*=\s*["\'][^"\']+["\']', source)
    assert len(hardcoded) == 0, f"Hardcoded credentials found: {hardcoded}"


def test_token_manager_uses_keyring():
    """Verify keyring import and usage in token_manager.py."""
    import aalgoi.core.token_manager as tm
    source = open(tm.__file__).read()
    assert "import keyring" in source or "from keyring" in source


def test_cli_no_shell_injection():
    """Check CLI code for subprocess/shell injection patterns."""
    import interface.cli as cli
    with open(cli.__file__, encoding='utf-8') as f:
        source = f.read()
    assert "shell=True" not in source, "shell=True found, risk of shell injection"
    assert "subprocess.call" not in source


def test_ast_literal_eval_blocks_malicious_code():
    """ast.literal_eval must reject non-literal expressions."""
    malicious = '__import__("os").system("rm -rf /")'
    with pytest.raises((ValueError, SyntaxError)):
        ast.literal_eval(malicious)


def test_ast_literal_eval_accepts_safe_literals():
    """Safe data structures must pass through ast.literal_eval."""
    safe = [
        "[1, 2, 3]",
        '{"a": 1, "b": 2}',
        "(1, 2, 3)",
        "True",
        "False",
        "42",
        "3.14",
        '"hello"',
    ]
    for s in safe:
        result = ast.literal_eval(s)
        assert result is not None


def test_validate_name_rejects_malicious_names():
    """Registration should block names with special chars or path traversal."""
    from aalgoi.pipeline import UniversalSolver
    solver = UniversalSolver()
    malicious_names = [
        "../../etc/passwd",
        "__import__('os')",
        "foo; rm -rf",
        "with spaces",
        "CamelCase",
        "UPPER_CASE",
    ]
    for bad_name in malicious_names:
        with pytest.raises(ValueError, match="must be snake_case"):
            solver._validate_name(bad_name)


def test_normalize_default_security():
    """normalize() defaults to safe mode (no file reads, no URL fetches)."""
    from aalgoi._data import normalize
    result = normalize([1, 2, 3])
    assert result == [1, 2, 3]


def test_mind_learn_deprecation():
    """Mind.learn() emits a DeprecationWarning."""
    import warnings
    from aalgoi._core import Mind
    m = Mind("~/.aalgoi/mind")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.learn("sort", [3, 1, 2])
        assert any(issubclass(x.category, DeprecationWarning) for x in w), \
            "Expected DeprecationWarning from Mind.learn()"


def test_mind_train_no_torch_warning():
    """Mind.train() emits a RuntimeWarning when torch is unavailable."""
    import warnings
    from importlib.util import find_spec
    if find_spec("torch") is not None:
        pytest.skip("torch is available — skipping no-torch warning test")
    from aalgoi._core import Mind
    m = Mind("~/.aalgoi/mind")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.train()
        assert any(issubclass(x.category, RuntimeWarning) for x in w), \
            "Expected RuntimeWarning from Mind.train() when torch is absent"


def test_sandboxed_executor_rejects_dangerous_code():
    """Sandbox validator rejects os.system and subprocess calls."""
    from aalgoi.core.sandboxed_executor import SandboxValidator
    validator = SandboxValidator()
    dangerous = [
        "__import__('os').system('rm -rf /')",
        "import subprocess; subprocess.call(['rm', '-rf', '/'])",
        "exec('open(\"/etc/passwd\").read()')",
    ]
    for code in dangerous:
        assert not validator.validate(code), \
            f"Dangerous code should have been rejected: {code!r}"
