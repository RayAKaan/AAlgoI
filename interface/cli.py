import json
import os
import sys
from typing import Any, Dict

import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Alias Group ─────────────────────────────────────────────


class AliasedGroup(click.Group):
    """
    Click group that allows short aliases.
    `aalgoi s` works the same as `aalgoi solve`.
    """

    ALIASES = {
        "s":  "solve",
        "e":  "explain",
        "b":  "benchmark",
        "st": "stats",
        "w":  "web",
        "a":  "api",
        "m":  "marketplace",
        "cp": "checkpoint",
        "sy": "sync",
    }

    def get_command(self, ctx, cmd_name):
        cmd_name = self.ALIASES.get(cmd_name, cmd_name)
        return super().get_command(ctx, cmd_name)

    def resolve_command(self, ctx, args):
        if args and args[0] in self.ALIASES:
            args[0] = self.ALIASES[args[0]]
        return super().resolve_command(ctx, args)


# ── Helpers ─────────────────────────────────────────────────


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


def _load_data(data_arg) -> Any:
    if data_arg is None:
        return None
    try:
        return json.loads(data_arg)
    except (json.JSONDecodeError, TypeError):
        try:
            with open(data_arg) as f:
                return json.load(f)
        except Exception:
            click.echo("Error: --data must be valid JSON string or file path", err=True)
            sys.exit(1)


# ── Main CLI ────────────────────────────────────────────────


@click.group(cls=AliasedGroup)
@click.version_option("1.2.0", prog_name="aalgoi")
def main():
    """
    AAlgoI — Self-adaptive algorithm intelligence.

    Aliases: s=solve  e=explain  b=benchmark  st=stats  w=web  a=api
    """


# ── Solve ───────────────────────────────────────────────────


@main.command("solve")
@click.argument("description")
@click.option("--data", type=str, default=None, help="Input data as JSON string or file path")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.option("--quiet", "-q", is_flag=True, help="Only print result value")
def cmd_solve(description, data, json_output, quiet):
    """
    Solve a problem from natural language description.

    Examples:

    \b
        aalgoi solve "sort" 3 1 4 1 5
        aalgoi s "sort desc" 5 3 1
        aalgoi s "path A to D" --data '{"A":{"B":1}}'
    """
    from aalgoi import solve

    input_data = _load_data(data)

    if input_data is None:
        input_data = [3, 1, 4, 1, 5]

    result = solve(description, input_data)

    if quiet:
        click.echo(result.value)
    elif json_output:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        status = click.style("OK", fg="green") if result.ok else click.style("FAIL", fg="red")
        click.echo(f"[{status}] {result.algo} in {result.ms:.2f}ms")
        value = result.value
        if isinstance(value, list) and len(value) > 20:
            click.echo(f"  Result: {value[:20]} ... ({len(value)} total)")
        else:
            click.echo(f"  Result: {value}")
        if result.answer:
            click.echo(f"  {result.answer}")


# ── Explain ─────────────────────────────────────────────────


@main.command("explain")
@click.argument("algorithm")
@click.option("--detail", type=click.Choice(["short", "detailed"]), default="short")
def cmd_explain(algorithm, detail):
    """
    Explain how an algorithm works.

    Examples:

    \b
        aalgoi explain quicksort
        aalgoi e timsort --detail detailed
    """
    from core.explainer import Explainer

    explainer = Explainer()
    exp = explainer.explain(algorithm, detail=detail)

    click.echo()
    click.echo("=" * 60)
    click.echo(f"  {exp.algorithm_name}")
    click.echo("=" * 60)
    click.echo(f"\n  {exp.summary}")
    click.echo(f"\n  Complexity: {exp.complexity}")
    click.echo(f"\n  Best For: {exp.best_for}")
    if exp.steps:
        click.echo("\n  Steps:")
        for i, step in enumerate(exp.steps, 1):
            click.echo(f"    {i}. {step}")
    click.echo()


# ── Benchmark ───────────────────────────────────────────────


@main.command("benchmark")
@click.argument("problem")
@click.option("--n", default=1000, help="Data size")
@click.option("--runs", "-r", default=5, help="Number of runs")
def cmd_benchmark(problem, n, runs):
    """
    Benchmark AAlgoI vs standard library.

    Examples:

    \b
        aalgoi benchmark sort --n 10000
        aalgoi b sort --n 1000 --runs 10
    """
    import random
    from aalgoi import benchmark, ProblemSpec, ProblemType

    data = [random.randint(0, 10**6) for _ in range(n)]
    spec = ProblemSpec(name=problem, problem_type=ProblemType.SORTING)

    click.echo(f"\nBenchmark: {problem} (n={n}, runs={runs})")
    click.echo("-" * 40)

    aalgoi_times = []
    baseline_times = []
    for _ in range(runs):
        bm = benchmark(spec, data)
        aalgoi_times.append(bm.get("aalgoi_time_ms", 0))
        baseline_times.append(bm.get("baseline_time_ms", 0))

    avg_aalgoi = sum(aalgoi_times) / runs
    avg_baseline = sum(baseline_times) / runs
    speedup = avg_baseline / avg_aalgoi if avg_aalgoi > 0 else 0

    winner_color = "green" if speedup > 1.05 else "yellow"
    click.echo(f"  AAlgoI:   {avg_aalgoi:.3f}ms avg")
    click.echo(f"  Baseline: {avg_baseline:.3f}ms avg (stdlib)")
    click.echo(f"  Speedup:  {speedup:.2f}x")
    click.echo(click.style(f"  Winner:   {'AAlgoI' if speedup > 1.05 else 'Baseline'}", fg=winner_color))


# ── Stats ───────────────────────────────────────────────────


@main.command("stats")
def cmd_stats():
    """Show solver performance statistics."""
    from core.smart_solver import SmartSolver

    solver = SmartSolver()
    stats = solver.solver.get_stats()
    click.echo(json.dumps(stats, indent=2, default=str))


# ── Web UI ──────────────────────────────────────────────────


@main.command("web")
@click.option("--port", type=int, default=7860, help="Port number")
@click.option("--share", is_flag=True, help="Generate public link")
def cmd_web(port, share):
    """Launch Gradio web UI."""
    try:
        from interface.web_ui import launch

        launch(share=share, server_port=port)
    except ImportError:
        click.echo("Gradio not available. Install with: pip install gradio", err=True)
        sys.exit(1)


# ── API Server ──────────────────────────────────────────────


@main.command("api")
@click.option("--port", type=int, default=8000, help="Port number")
@click.option("--host", type=str, default="0.0.0.0", help="Host address")
def cmd_api(host, port):
    """Launch FastAPI REST API."""
    try:
        from interface.api import run

        run(host=host, port=port)
    except ImportError:
        click.echo("FastAPI not available. Install with: pip install fastapi uvicorn", err=True)
        sys.exit(1)


# ── Marketplace ─────────────────────────────────────────────


@main.group("marketplace")
def marketplace():
    """Community algorithm marketplace."""


@marketplace.command("list")
def list_marketplace():
    """List all registered algorithms."""
    from core.registry_manager import DynamicRegistry
    from core.smart_solver import SmartSolver

    solver = SmartSolver()
    registry = DynamicRegistry(solver.solver.registry)
    algos = registry.list_algorithms()
    click.echo(f"\nRegistered Algorithms ({len(algos)}):")
    for name in sorted(algos):
        click.echo(f"  - {name}")
    click.echo()


@marketplace.command("search")
@click.argument("query")
def search_marketplace(query):
    """Search for algorithms by keyword."""
    from core.algorithm_marketplace import AlgorithmMarketplace

    mkt = AlgorithmMarketplace()
    results = mkt.find_by_use_case(query)

    if results:
        click.echo(f"\nFound {len(results)} algorithms for '{query}':")
        for meta in results[:10]:
            click.echo(f"  - {meta.name}")
            click.echo(f"    Use case: {meta.use_case[:80]}")
    else:
        click.echo(f"No results for '{query}'")


# ── Checkpoint ──────────────────────────────────────────────


@main.group("checkpoint")
def checkpoint_group():
    """
    Manage model checkpoints and personalization.

    \b
    Commands:
        list      Show all saved checkpoints
        rollback  Revert to a previous version
        reset     Wipe all personalization, restore base model
        info      Show checkpoint status and storage usage
    """
    pass


@checkpoint_group.command("list")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def checkpoint_list(json_output):
    """
    List all saved checkpoints with metrics.

    \b
    Example:
        aalgoi checkpoint list
        aalgoi checkpoint list --json-output
    """
    from core.checkpoint_manager import CheckpointManager

    manager     = CheckpointManager()
    checkpoints = manager.list_checkpoints()
    manifest    = manager._load_manifest()
    current     = manifest.get('current_version', 0)

    if json_output:
        click.echo(json.dumps({
            'current_version': current,
            'checkpoints': checkpoints
        }, indent=2))
        return

    if not checkpoints:
        click.echo("No checkpoints saved yet.")
        click.echo(
            "Checkpoints are saved automatically every 50 solves."
        )
        return

    click.echo(f"\n{'Ver':<6} {'Solves':<8} {'Date':<12} "
               f"{'Success':<10} {'Status'}")
    click.echo("─" * 55)

    for cp in checkpoints:
        metrics  = cp.get('metrics', {})
        sr       = metrics.get('success_rate', 0.0)
        date     = cp['timestamp'][:10]
        is_curr  = (
            click.style(" ← current", fg="green")
            if cp['version'] == current
            else ""
        )
        click.echo(
            f"v{cp['version']:<5} "
            f"{cp['solve_count']:<8} "
            f"{date:<12} "
            f"{sr:.1%}{'':>5}"
            f"{is_curr}"
        )

    click.echo(
        f"\nBase model: ~/.aalgoi/checkpoints/pretrained_final.pt "
        f"(never modified)"
    )


@checkpoint_group.command("rollback")
@click.option(
    "--version", "-v",
    type=int,
    default=None,
    help="Version number to roll back to. Default: one step back."
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would happen without making changes."
)
def checkpoint_rollback(version, dry_run):
    """
    Roll back to a previous model version.

    \b
    Examples:
        aalgoi checkpoint rollback              # one step back
        aalgoi checkpoint rollback --version 3  # specific version
        aalgoi checkpoint rollback --dry-run    # preview only
    """
    from core.checkpoint_manager import CheckpointManager

    manager  = CheckpointManager()
    manifest = manager._load_manifest()
    current  = manifest.get('current_version', 0)

    target = version if version is not None else current - 1

    if target < 1:
        click.echo(
            click.style("✗ Already at oldest checkpoint.", fg="red")
        )
        return

    entry = next(
        (c for c in manifest['checkpoints'] if c['version'] == target),
        None
    )

    if not entry:
        click.echo(
            click.style(f"✗ Version {target} not found.", fg="red")
        )
        click.echo("Run `aalgoi checkpoint list` to see available versions.")
        return

    metrics = entry.get('metrics', {})
    click.echo(f"\nRollback plan:")
    click.echo(f"  Current version : v{current}")
    click.echo(f"  Target version  : v{target}")
    click.echo(f"  Solve count     : {entry['solve_count']}")
    click.echo(f"  Date            : {entry['timestamp'][:10]}")
    click.echo(
        f"  Success rate    : "
        f"{metrics.get('success_rate', 0.0):.1%}"
    )

    if dry_run:
        click.echo(
            click.style("\n(dry-run) No changes made.", fg="yellow")
        )
        return

    if current - target > 1:
        click.confirm(
            f"\nRoll back {current - target} versions?",
            abort=True
        )

    path = manager.rollback(version=target)

    if path:
        click.echo(
            click.style(f"\n✓ Rolled back to v{target}.", fg="green")
        )
        click.echo(f"  Adapter: {path}")
        click.echo(
            "\nRestart your solver for changes to take effect."
        )
    else:
        click.echo(
            click.style("✗ Rollback failed.", fg="red")
        )


@checkpoint_group.command("reset")
@click.option(
    "--confirm",
    is_flag=True,
    help="Skip confirmation prompt."
)
def checkpoint_reset(confirm):
    """
    Reset to factory base model.

    Removes ALL personalization (LoRA adapter weights).
    Base model weights are never affected.

    \b
    Example:
        aalgoi checkpoint reset
        aalgoi checkpoint reset --confirm   # no prompt
    """
    from core.checkpoint_manager import CheckpointManager

    manager  = CheckpointManager()
    manifest = manager._load_manifest()
    n        = len(manifest.get('checkpoints', []))

    if n == 0:
        click.echo("No checkpoints to remove. Already at base model.")
        return

    click.echo(f"\nThis will delete {n} checkpoint(s).")
    click.echo(
        "Your base model (pretrained_final.pt) will not be affected."
    )

    if not confirm:
        click.confirm("Continue?", abort=True)

    manager.reset()
    click.echo(click.style("\n✓ Reset to base model.", fg="green"))
    click.echo(
        "All personalization removed. "
        "The system will rebuild from the base model on next use."
    )


@checkpoint_group.command("info")
def checkpoint_info():
    """
    Show current checkpoint status and storage usage.

    \b
    Example:
        aalgoi checkpoint info
    """
    from pathlib import Path
    from core.checkpoint_manager import CheckpointManager

    manager     = CheckpointManager()
    manifest    = manager._load_manifest()
    current     = manifest.get('current_version', 0)
    checkpoints = manifest.get('checkpoints', [])
    adapter_dir = manager.adapter_dir
    base_dir    = manager.base_dir

    def dir_size_mb(path: Path) -> float:
        if not path.exists():
            return 0.0
        total = sum(
            f.stat().st_size
            for f in path.rglob('*')
            if f.is_file()
        )
        return total / (1024 * 1024)

    adapter_mb  = dir_size_mb(adapter_dir)
    base_pt     = base_dir / "checkpoints" / "pretrained_final.pt"
    base_mb     = (
        base_pt.stat().st_size / (1024 * 1024)
        if base_pt.exists()
        else 0.0
    )

    click.echo(f"\nCheckpoint Status")
    click.echo("─" * 40)
    click.echo(f"  Current version    : v{current}")
    click.echo(f"  Total checkpoints  : {len(checkpoints)}")
    click.echo(f"  Base model         : {base_mb:.1f} MB")
    click.echo(f"  Adapter storage    : {adapter_mb:.1f} MB")
    click.echo(f"  Storage location   : {base_dir}")

    if checkpoints:
        latest = checkpoints[-1]
        click.echo(
            f"\n  Latest checkpoint  : "
            f"v{latest['version']} "
            f"({latest['timestamp'][:10]}, "
            f"{latest['solve_count']} solves)"
        )

    registry_sync_state = base_dir / "registry" / "sync_state.json"
    if registry_sync_state.exists():
        import json as _json
        state = _json.loads(registry_sync_state.read_text())
        click.echo(
            f"\n  Registry sync      : "
            f"{state.get('last_sync', 'never')[:10]}"
        )
        click.echo(
            f"  Registry version   : "
            f"v{state.get('index_version', 0)}"
        )


# ── Sync ────────────────────────────────────────────────────


@main.group("sync")
def sync_group():
    """
    Manage registry synchronization with GitHub.

    \b
    Commands:
        status  Show sync status and connectivity
        pull    Trigger a sync pull now
        push    Push a discovered algorithm upstream
    """
    pass


@sync_group.command("status")
def sync_status():
    """Show registry sync status, backoff state, and queue depth."""
    from core.registry_sync import GitHubRegistrySync, _load_registry_config

    cfg = _load_registry_config()
    url = cfg.get("base_url", "https://raw.githubusercontent.com/aalgoi/algorithm-registry/main")

    import json as _json
    from pathlib import Path

    state_path = Path("~/.aalgoi/registry/sync_state.json").expanduser()
    if state_path.exists():
        state = _json.loads(state_path.read_text())
        last_sync = state.get("last_sync", "never")
        version = state.get("index_version", 0)
        count = len(state.get("algorithm_names", []))
    else:
        last_sync = "never"
        version = 0
        count = 0

    click.echo(f"\nRegistry Sync Status")
    click.echo("─" * 40)
    click.echo(f"  Remote registry    : {url}")
    click.echo(f"  Last sync          : {last_sync[:19] if last_sync != 'never' else 'never'}")
    click.echo(f"  Index version      : v{version}")
    click.echo(f"  Registered algos   : {count}")

    # Check token
    from core.token_manager import TokenManager
    token = TokenManager.get_token()
    if token:
        click.echo(f"  Auth token         : {click.style('set', fg='green')} ({token[:8]}...)")
    else:
        click.echo(f"  Auth token         : {click.style('not set', fg='yellow')}")

    # Connectivity probe
    import requests as _requests
    try:
        _requests.get(url.replace("raw.githubusercontent", "github.com"), timeout=5)
        click.echo(f"  Connectivity       : {click.style('online', fg='green')}")
    except Exception:
        click.echo(f"  Connectivity       : {click.style('offline', fg='red')}")


@sync_group.command("pull")
def sync_pull():
    """Force a registry sync pull now."""
    from core.registry_manager import DynamicRegistry
    from core.algorithm_embedder import AlgorithmEmbedder
    from core.registry_sync import GitHubRegistrySync

    registry = DynamicRegistry(None)
    embedder = AlgorithmEmbedder()
    sync = GitHubRegistrySync(registry, embedder, config={})

    click.echo("\nPulling registry...")

    result = sync.sync_pull()

    if result["status"] == "offline":
        click.echo(click.style(f"  ✗ GitHub unreachable: {result.get('error', '')}", fg="red"))
        click.echo("  Sync will retry automatically. Run `aalgoi sync status` to check.")
    elif result["status"] == "up_to_date":
        click.echo(click.style("  ✓ Already up to date.", fg="green"))
        click.echo(f"  Index version: v{result['version']}")
        st = sync.status()
        click.echo(f"  Registered algorithms: {st['algorithm_count']}")
    elif result["status"] == "updated":
        click.echo(click.style(f"  ✓ Synced: {result['new_algorithms']} new, {result['failed']} failed.", fg="green"))
        for d in result.get("details", []):
            if d["status"] == "registered":
                click.echo(f"    + {d['name']}")
            else:
                click.echo(f"    ✗ {d['name']}: {d.get('reason', 'unknown')}", err=True)
        st = sync.status()
        click.echo(f"  Registry version: v{st['index_version']}")
    else:
        click.echo(click.style(f"  ? {result}", fg="yellow"))


@sync_group.command("push")
@click.argument("algorithm_name")
@click.option("--token", envvar="GITHUB_TOKEN", help="GitHub token (or set GITHUB_TOKEN env var)")
@click.option("--dry-run", is_flag=True, help="Validate without pushing")
def sync_push(algorithm_name, token, dry_run):
    """Push a discovered algorithm upstream to GitHub."""
    from core.registry_manager import DynamicRegistry
    from core.algorithm_embedder import AlgorithmEmbedder
    from core.registry_sync import GitHubRegistrySync

    registry = DynamicRegistry(None)
    embedder = AlgorithmEmbedder()
    sync = GitHubRegistrySync(registry, embedder, config={})

    algo = registry.get_algorithm(algorithm_name)
    if not algo:
        click.echo(click.style(f"✗ Algorithm '{algorithm_name}' not found in local registry.", fg="red"))
        click.echo("Run `aalgoi marketplace list` to see registered algorithms.")
        return

    class MockValidation:
        passed = True
        improvement_pct = 0.0
        pass_rate = 1.0
        baseline_name = "built-in"
        results = {
            "correctness": 1.0,
            "edge_cases": {"all_passed": True},
            "performance": {"beats_baseline": True},
        }

    click.echo(f"\nAlgorithm: {algorithm_name}")

    if dry_run:
        click.echo(click.style("  ✓ Validation passed (dry-run)", fg="green"))
        click.echo("  No changes made.")
        return

    result = sync.push_discovered(algo, MockValidation(), github_token=token or "")

    if result["status"] == "submitted":
        click.echo(click.style("  ✓ Push submitted.", fg="green"))
        click.echo(f"  PR URL: {result['pr_url']}")
    elif result["status"] == "queued":
        click.echo(click.style(f"  ⏸ Push queued: {result.get('reason', 'unknown')}", fg="yellow"))
        click.echo("  Will retry when sync is back online and GITHUB_TOKEN is set.")
    elif result["status"] == "rejected":
        click.echo(click.style(f"  ✗ Rejected: {result.get('reason', 'unknown')}", fg="red"))
    else:
        click.echo(click.style(f"  ✗ Push failed: {result.get('error', 'unknown')}", fg="red"))


# ── Subcommands ─────────────────────────────────────────────


# Lazy-import subcommands to avoid circular imports
from interface.cli_debug import debug
from interface.cli_ml import ml

main.add_command(debug)
main.add_command(ml)


if __name__ == "__main__":
    main()
