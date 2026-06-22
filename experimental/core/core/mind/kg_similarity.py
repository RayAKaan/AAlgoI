from typing import Any


def compute_problem_similarity(
    sig_a: dict[str, Any],
    sig_b: dict[str, Any],
) -> float:
    score = 0.0

    if sig_a.get("domain") == sig_b.get("domain"):
        if sig_a.get("domain") and sig_a.get("domain") != "unknown":
            score += 0.35

    struct_a = sig_a.get("hidden_structure", "unknown")
    struct_b = sig_b.get("hidden_structure", "unknown")
    if struct_a == struct_b and struct_a != "unknown":
        score += 0.30

    if sig_a.get("optimization_goal") == sig_b.get("optimization_goal"):
        if sig_a.get("optimization_goal"):
            score += 0.20

    n_a = sig_a.get("constraint_n", 0)
    n_b = sig_b.get("constraint_n", 0)
    if n_a > 0 and n_b > 0:
        import math
        log_ratio = abs(math.log10(max(n_a, 1)) - math.log10(max(n_b, 1)))
        constraint_sim = max(0.0, 1.0 - log_ratio / 3.0)
        score += 0.15 * constraint_sim

    return min(score, 1.0)


def extract_signature_dict(
    domain: str,
    optimization_goal: str,
    hidden_structure: str,
    constraint_n: int = 0,
) -> dict[str, Any]:
    return {
        "domain": domain,
        "optimization_goal": optimization_goal,
        "hidden_structure": hidden_structure,
        "constraint_n": constraint_n,
    }
