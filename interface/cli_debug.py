"""
Debug and Visualization CLI for AAlgoI.
Shows context analysis, RL policy distribution, and reasoning chain.
"""

import sys
import os

import click
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@click.group()
def debug():
    """Debug and visualization tools."""
    pass


@debug.command()
@click.argument("problem")
@click.option("--data", default=None, help="JSON data or file path")
@click.option("--json-output", "json_flag", is_flag=True, help="Output as JSON")
def visualize(problem, data, json_flag):
    """Visualize how AAlgoI solves a problem."""
    import json as json_module
    from aalgoi.core.smart_solver import SmartSolver

    solver = SmartSolver()

    if data:
        spec = solver.parser.parse(problem)
        data_parsed = _load_data_arg(data)
    else:
        spec = solver.parser.parse(problem)
        data_parsed = None

    click.echo()
    click.echo("=" * 60)
    click.echo("  CONTEXT ANALYSIS")
    click.echo("=" * 60)
    click.echo(f"  Problem Type: {spec.problem_type.value}")
    click.echo(f"  Description: {problem[:80]}")

    context = solver.solver.context_engine.analyze(
        data_parsed or [],
        spec.problem_type.value,
    )

    dp = context.get("data_profile", {})
    click.echo(f"  Data Size: {dp.get('size', 'N/A')}")
    click.echo(f"  Data Type: {dp.get('type', 'N/A')}")
    if dp.get("patterns"):
        click.echo(f"  Patterns: {dp['patterns']}")

    click.echo()
    click.echo("=" * 60)
    click.echo("  REASONING CHAIN")
    click.echo("=" * 60)

    result = solver.solver.solve(spec, data_parsed)

    click.echo(f"  Selected Algorithm: {result.get('algorithm', 'unknown')}")
    click.echo(f"  Success: {result.get('success', False)}")
    click.echo(f"  Time: {result.get('time_ms', 0):.2f} ms")

    sel = result.get("selection", {})
    click.echo(f"  Strategy: {sel.get('synthesis_strategy', 'unknown')}")
    click.echo(f"  Confidence: {sel.get('confidence', 0):.2f}")

    if json_flag:
        click.echo(json_module.dumps({
            "context": {
                "problem_type": spec.problem_type.value,
                "data_size": dp.get("size"),
                "patterns": dp.get("patterns"),
            },
            "result": {
                "algorithm": result.get("algorithm"),
                "success": result.get("success"),
                "time_ms": result.get("time_ms"),
                "confidence": sel.get("confidence"),
            },
        }, indent=2))


def _load_data_arg(data_arg):
    import json
    try:
        return json.loads(data_arg)
    except json.JSONDecodeError:
        try:
            with open(data_arg, "r") as f:
                return json.load(f)
        except Exception:
            return None
