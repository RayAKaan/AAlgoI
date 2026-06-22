from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from aalgoi import __version__
from aalgoi.api import Mind, solve


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aalgoi",
        description="aalgoi — Hybrid Algorithmic AI",
    )
    parser.add_argument("--path", "-p", default="~/.aalgoi/mind", help="Mind persistence path")
    parser.add_argument("--version", "-v", action="store_true", help="Show version")

    sub = parser.add_subparsers(dest="command", help="Commands")

    solve_p = sub.add_parser("solve", help="Solve an algorithmic problem")
    solve_p.add_argument("problem", help="Problem description (e.g. 'sort [3,1,2]')")
    solve_p.add_argument("--data", "-d", help="Input data (JSON)")
    solve_p.add_argument("--file", "-f", help="Input data file")
    solve_p.add_argument("--mode", "-m", default="deterministic", choices=["deterministic", "learned", "experimental"])
    solve_p.add_argument("--explain", "-e", action="store_true")

    sub.add_parser("status", help="Show mind status")

    train_p = sub.add_parser("train", help="Train the learned selector")
    train_p.add_argument("--mode", default="bandit", choices=["bandit", "supervised"])
    train_p.add_argument("--suite", default="core-v1", help="Benchmark suite to train on")

    bench_p = sub.add_parser("benchmark", help="Run benchmark suite")
    bench_p.add_argument("--suite", default="core-v1")
    bench_p.add_argument("--verbose", "-v", action="store_true")

    kg_p = sub.add_parser("kg", help="Knowledge Graph commands")
    kg_p.add_argument("action", choices=["inspect", "stats"])
    kg_p.add_argument("--algorithm", "-a", help="Algorithm name to inspect")

    sub.add_parser("registry", help="List registered algorithms")
    sub.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.version:
        print(f"aalgoi {__version__}")
        return

    if args.command is None:
        parser.print_help()
        return

    mind_path = Path(args.path).expanduser()

    if args.command == "solve":
        _cmd_solve(args)
    elif args.command == "status":
        _cmd_status(mind_path)
    elif args.command == "train":
        _cmd_train(args, mind_path)
    elif args.command == "benchmark":
        _cmd_benchmark(args, mind_path)
    elif args.command == "kg":
        _cmd_kg(args)
    elif args.command == "registry":
        _cmd_registry()
    elif args.command == "version":
        print(f"aalgoi {__version__}")


def _cmd_solve(args: argparse.Namespace) -> None:
    data = None
    if args.data:
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError:
            data = args.data
    elif args.file:
        from aalgoi.data.file_loaders import load_file
        data = load_file(args.file)

    result = solve(args.problem, data, mode=args.mode)

    if result.ok:
        if isinstance(result.output, (list, dict)):
            print(json.dumps(result.output, indent=2))
        else:
            print(result.output)
    else:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)

    if args.explain:
        print()
        lines = [
            f"Algorithm: {result.algorithm or 'N/A'}",
            f"Validated: {result.validated}",
            f"Time: {result.time_ms:.1f}ms",
            f"Confidence: {result.confidence:.2f}",
            f"Mode: {result.mode}",
        ]
        print("\n".join(lines))


def _cmd_status(mind_path: Path) -> None:
    mind = Mind(str(mind_path))
    print(mind.status())


def _cmd_train(args: argparse.Namespace, mind_path: Path) -> None:
    mind = Mind(str(mind_path))
    stats = mind.train(mode=args.mode, suite=args.suite)
    if isinstance(stats, dict) and stats.get("status") == "unavailable":
        print(f"Training unavailable: {stats.get('error', 'install aalgoi[rl] for RL training')}")
    else:
        print(f"Training completed: {stats}")


def _cmd_benchmark(args: argparse.Namespace, mind_path: Path) -> None:
    mind = Mind(str(mind_path))
    report = mind.benchmark(suite=args.suite)
    print(report)
    if args.verbose and hasattr(report, "data") and report.data.by_task:
        print("\nBy task:")
        for task_key, d in sorted(report.data.by_task.items()):
            pct = d["passed"] / max(d["total"], 1) * 100
            print(f"  {task_key}: {d['passed']}/{d['total']} ({pct:.0f}%)")


def _cmd_kg(args: argparse.Namespace) -> None:
    from aalgoi.kg.graph import KnowledgeGraph
    from aalgoi.kg.seed import seed_from_registry
    from aalgoi.algorithms.registry import get_registry

    kg = KnowledgeGraph()
    seed_from_registry(kg, get_registry())

    if args.action == "stats":
        stats = kg.stats()
        print(f"Nodes: {stats['nodes']}")
        print(f"Edges: {stats['edges']}")
        print(f"Algorithms: {stats['algorithms']}")
        print(f"Tasks: {stats['tasks']}")
    elif args.action == "inspect":
        if args.algorithm:
            expl = kg.explain(args.algorithm)
            if expl:
                print(expl)
            else:
                print(f"Algorithm '{args.algorithm}' not found in KG")
        else:
            for name in get_registry().get_names():
                expl = kg.explain(name)
                if expl:
                    print(f"  {expl}")


def _cmd_registry() -> None:
    from aalgoi.algorithms.registry import get_registry
    reg = get_registry()
    print(f"Algorithm Registry ({len(reg)} algorithms):")
    for name in sorted(reg.get_names()):
        spec = reg.get_spec(name)
        if spec:
            print(f"  {name} ({spec.task.value}) — {spec.complexity.time} / {spec.complexity.space}")
