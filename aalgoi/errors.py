class AAlgoIError(Exception):
    """Base error for all aalgoi exceptions."""


class NoAlgorithmFound(AAlgoIError):
    """No registered algorithm could handle the given problem."""


class ValidationFailed(AAlgoIError):
    """Algorithm output failed validation."""


class UnsafeCode(AAlgoIError):
    """Code was rejected by the security sandbox."""


class OptionalDependencyMissing(AAlgoIError):
    """An optional dependency is not installed (install with extras)."""

    def __init__(self, feature: str, package: str) -> None:
        self.feature = feature
        self.package = package
        super().__init__(f"'{feature}' requires '{package}' — install with: pip install aalgoi[{feature}]")


class ParseError(AAlgoIError):
    """Could not parse the problem description into a ProblemSpec."""
