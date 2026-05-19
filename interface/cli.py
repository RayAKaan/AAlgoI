import json
import os
import sys
from typing import Any, Dict, Optional

import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.problem_spec import ProblemSpec, ProblemType
from core.smart_solver import SmartSolver
from core.explainer import Explainer
from core.benchmarker import Benchmarker
from interface.nl_parser import parse_description, parse_solve_input
from interface.cli_ml import ml
from interface.cli_debug import debug


def _serialize_for_json(obj: Any) -> Any:
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        if len(obj) > 20:
            return list(obj[:20]) + ["..."]
        return list(obj)
    if isinstance(obj, dict):
        return {str(k): _serialize_for_json(v) for k, v in obj.items()}
    return str(obj)


def _print_result(result: Dict[str, Any]):
    click.echo()
    click.echo("=" * 60)
    click.echo("  Result")
    click.echo("=" * 60)
    r = result.get("result")
    if isinstance(r, list) and len(r) > 20:
        click.echo(f"  Output: {r[:20]} ... ({len(r)} total)")
    else:
        click.echo(f"  Output: {r}")
    click.echo(f"  Success: {result.get('success', False)}")
    click.echo(f"  Time: {result.get('time_ms', 0):.2f} ms")
    click.echo(f"  Pipeline: {' -> '.join(result.get('pipeline', []))}")

    sel = result.get("selection", {})
    click.echo(f"  Strategy: {sel.get('synthesis_strategy', 'unknown')}")
    click.echo(f"  Confidence: {sel.get('confidence', 0):.2f}")

    validation = result.get("validation", [])
    if validation:
        click.echo()
        click.echo("  Validation:")
        for v in validation:
            status = click.style("[OK]", fg="green") if v.get("passed") else click.style("[FAIL]", fg="red")
            click.echo(f"    {status} {v['algorithm']}")

    explanations = result.get("explanation", [])
    if explanations:
        click.echo()
        click.echo("  Explanation:")
        for exp in explanations:
            click.echo(f"    {exp.algorithm_name}: {exp.summary[:100]}...")
    click.echo()


def _print_json_result(result: Dict[str, Any]):
    output = {
        "success": result.get("success"),
        "result": _serialize_for_json(result.get("result")),
        "time_ms": result.get("time_ms"),
        "pipeline": result.get("pipeline"),
        "strategy": result.get("selection", {}).get("synthesis_strategy"),
        "confidence": result.get("selection", {}).get("confidence"),
    }
    click.echo(json.dumps(output, indent=2))


def _load_data(data_arg: Optional[str]) -> Any:
    if data_arg is None:
        return None
    try:
        return json.loads(data_arg)
    except json.JSONDecodeError:
        try:
            with open(data_arg, "r") as f:
                return json.load(f)
        except Exception:
            click.echo("Error: --data must be valid JSON string or file path", err=True)
            sys.exit(1)


# ============================================
# MAIN CLI
# ============================================

@click.group(invoke_without_command=False)
@click.version_option("1.1.0", prog_name="aalgoi")
def main():
    """AAlgoI - Self-Adaptive Algorithm Intelligence\n\n
    Universal problem solver with RL-discovered algorithms.
    """
    pass


# ============================================
# SOLVE COMMANDS
# ============================================

@main.command()
@click.argument("description")
@click.option("--data", type=str, default=None, help="Input data as JSON string or file path")
@click.option("--llm", is_flag=True, help="Use LLM for synthesis")
@click.option("--json-output", "json_flag", is_flag=True, help="Output as JSON")
def solve(description, data, llm, json_flag):
    """Solve a problem from natural language description."""
    spec = parse_description(description)
    data = _load_data(data)

    if data is None:
        _, data = parse_solve_input(description)
    if data is None:
        data = [3, 1, 4, 1, 5]

    solver = SmartSolver()
    result = solver.solver.solve(spec, data, use_llm=llm)

    if json_flag:
        _print_json_result(result)
    else:
        _print_result(result)


@main.command("solve-spec")
@click.argument("spec_file", type=click.Path(exists=True))
@click.option("--data", type=str, default=None, help="Input data as JSON string or file path")
@click.option("--llm", is_flag=True, help="Use LLM for synthesis")
@click.option("--json-output", "json_flag", is_flag=True, help="Output as JSON")
def solve_spec(spec_file, data, llm, json_flag):
    """Solve a problem from a JSON ProblemSpec file."""
    try:
        with open(spec_file, "r") as f:
            spec_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        click.echo(f"Error reading spec file: {e}", err=True)
        sys.exit(1)

    spec = ProblemSpec.from_dict(spec_dict)
    data = _load_data(data)
    if data is None:
        data = [3, 1, 2]

    solver = SmartSolver()
    result = solver.solver.solve(spec, data, use_llm=llm)

    if json_flag:
        _print_json_result(result)
    else:
        _print_result(result)


# ============================================
# EXPLAIN
# ============================================

@main.command()
@click.argument("algorithm")
@click.option("--detail", type=click.Choice(["short", "detailed"]), default="short")
@click.option("--json-output", "json_flag", is_flag=True, help="Output as JSON")
def explain(algorithm, detail, json_flag):
    """Explain how an algorithm works."""
    explainer = Explainer()
    exp = explainer.explain(algorithm, detail=detail)

    if json_flag:
        click.echo(json.dumps({
            "algorithm": exp.algorithm_name,
            "summary": exp.summary,
            "complexity": exp.complexity,
            "steps": exp.steps,
            "best_for": exp.best_for,
            "source": exp.source
        }, indent=2))
    else:
        click.echo()
        click.echo("=" * 60)
        click.echo(f"  {exp.algorithm_name}")
        click.echo("=" * 60)
        click.echo(f"\n  {exp.summary}")
        click.echo(f"\n  Complexity: {exp.complexity}")
        click.echo(f"\n  Best For: {exp.best_for}")
        if exp.steps:
            click.echo(f"\n  Steps:")
            for i, step in enumerate(exp.steps, 1):
                click.echo(f"    {i}. {step}")
        click.echo()


# ============================================
# STATS
# ============================================

@main.command()
@click.option("--json-output", "json_flag", is_flag=True, help="Output as JSON")
def stats(json_flag):
    """Show solver statistics."""
    solver = SmartSolver()
    s = solver.solver.get_stats()

    if json_flag:
        click.echo(json.dumps(s, indent=2))
    else:
        click.echo()
        click.echo("=" * 60)
        click.echo("  AAlgoI Statistics")
        click.echo("=" * 60)
        click.echo(f"  Total solves: {s.get('total_solves', 0)}")
        vstats = s.get("validator", {})
        click.echo(f"  Validations: {vstats.get('total_validations', 0)} total, "
                    f"{vstats.get('failed', 0)} failed")
        click.echo()


# ============================================
# BENCHMARK
# ============================================

@main.command()
@click.argument("problem")
@click.option("--data", default=None, help="JSON data or file path")
def benchmark(problem, data):
    """Benchmark AAlgoI vs standard library."""
    from core.smart_solver import SmartSolver

    solver = SmartSolver()
    bench = Benchmarker(solver.solver)
    spec = solver.parser.parse(problem)

    data = _load_data(data) or [5, 3, 1, 4, 2]
    result = bench.compare(spec, data)

    click.echo()
    click.echo("=" * 50)
    click.echo("  BENCHMARK RESULTS")
    click.echo("=" * 50)
    click.echo(f"  AAlgoI:    {result['aalgoi_time_ms']:.2f}ms ({result['aalgoi_algorithm']})")
    click.echo(f"  Baseline:  {result['baseline_time_ms']:.2f}ms ({result['baseline_algorithm']})")
    click.echo(f"  Speedup:   {result['speedup_factor']:.2f}x")
    click.echo(f"  Winner:    {result['winner']}")
    click.echo("=" * 50)


# ============================================
# MARKETPLACE
# ============================================

@main.group()
def marketplace():
    """Community algorithm marketplace."""
    pass


@marketplace.command("list")
@click.option("--json-output", "json_flag", is_flag=True, help="Output as JSON")
def list_marketplace(json_flag):
    """List all registered algorithms."""
    from core.registry_manager import DynamicRegistry

    solver = SmartSolver()
    registry = DynamicRegistry(solver.solver.registry)
    algos = registry.list_algorithms()

    if json_flag:
        click.echo(json.dumps(algos, indent=2))
    else:
        click.echo(f"\nRegistered Algorithms ({len(algos)}):")
        for name in sorted(algos):
            click.echo(f"  - {name}")
        click.echo()


@marketplace.command("search")
@click.argument("query")
def search_marketplace(query):
    """Search for algorithms by keyword or use case."""
    from core.algorithm_marketplace import AlgorithmMarketplace

    mkt = AlgorithmMarketplace()
    results = mkt.find_by_use_case(query)

    if results:
        click.echo(f"\nFound {len(results)} algorithms for '{query}':")
        for meta in results[:10]:
            click.echo(f"  - {meta.name}")
            click.echo(f"    Use case: {meta.use_case[:80]}")
            click.echo(f"    Reward: {meta.avg_reward:.2f}")
    else:
        click.echo(f"No results for '{query}'")

        try:
            import requests
            resp = requests.get(
                "https://api.aalgoi.org/marketplace/search",
                params={"q": query},
                timeout=3,
            )
            if resp.status_code == 200:
                remote = resp.json()
                click.echo(f"\nRemote marketplace found {len(remote)} results")
                for algo in remote[:5]:
                    click.echo(f"  - {algo.get('name', 'unknown')}")
        except Exception:
            click.echo("(Remote marketplace unavailable)")


# ============================================
# WEB / API
# ============================================

@main.command()
@click.option("--port", type=int, default=7860, help="Port number")
@click.option("--share", is_flag=True, help="Generate public link")
def web(port, share):
    """Launch Gradio web UI."""
    try:
        from interface.web_ui import launch
        launch(share=share, server_port=port)
    except ImportError:
        click.echo("Gradio not available. Install with: pip install gradio", err=True)
        sys.exit(1)


@main.command()
@click.option("--port", type=int, default=8000, help="Port number")
@click.option("--host", type=str, default="0.0.0.0", help="Host address")
def api(host, port):
    """Launch FastAPI REST API."""
    try:
        from interface.api import run
        run(host=host, port=port)
    except ImportError:
        click.echo("FastAPI not available. Install with: pip install fastapi uvicorn", err=True)
        sys.exit(1)


# ============================================
# REGISTER SUBCOMMANDS
# ============================================

main.add_command(ml)
main.add_command(debug)
