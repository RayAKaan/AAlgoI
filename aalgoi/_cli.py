"""
Command-line interface for aalgoi.

Usage:
    aalgoi solve "sort the array" --data '[3,1,4,1,5]'
    aalgoi solve "sort" --file data.json --explain
    aalgoi status
    aalgoi train --epochs 10
    aalgoi benchmark
    aalgoi checkpoint list
    aalgoi checkpoint save
    aalgoi rollback --target last_good
    aalgoi share
    aalgoi receive
    aalgoi version
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aalgoi",
        description="aalgoi — Algorithmic AI that learns, reasons, and discovers",
    )
    parser.add_argument(
        "--path", "-p",
        default="~/.aalgoi/mind",
        help="Mind persistence path (default: ~/.aalgoi/mind)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── solve ────────────────────────────────────────────────────────
    solve_p = subparsers.add_parser("solve", help="Solve a problem")
    solve_p.add_argument("problem", help="Problem description")
    solve_p.add_argument("--data", "-d", help="Input data (JSON)")
    solve_p.add_argument("--file", "-f", help="Input data file (JSON/CSV/etc)")
    solve_p.add_argument("--explain", "-e", action="store_true", help="Show explanation")
    solve_p.add_argument("--code", "-c", action="store_true", help="Show generated code")
    # ── status ───────────────────────────────────────────────────────
    subparsers.add_parser("status", help="Show mind status")

    # ── train ────────────────────────────────────────────────────────
    train_p = subparsers.add_parser("train", help="Bootstrap train the mind")
    train_p.add_argument("--epochs", type=int, default=10, help="Training epochs")
    train_p.add_argument("--augmentations", type=int, default=5, help="Data augmentations")

    # ── benchmark ────────────────────────────────────────────────────
    bench_p = subparsers.add_parser("benchmark", help="Run benchmark suite")
    bench_p.add_argument("--verbose", "-v", action="store_true")

    # ── checkpoint ───────────────────────────────────────────────────
    cp_p = subparsers.add_parser("checkpoint", help="Manage checkpoints")
    cp_p.add_argument("action", choices=["list", "save"], help="Checkpoint action")

    # ── rollback ─────────────────────────────────────────────────────
    rb_p = subparsers.add_parser("rollback", help="Rollback mind to safe state")
    rb_p.add_argument("--target", default="last_good", choices=["last", "last_good", "base"])

    # ── share ────────────────────────────────────────────────────────
    subparsers.add_parser("share", help="Show federated updates to share")

    # ── receive ──────────────────────────────────────────────────────
    subparsers.add_parser("receive", help="Import federated updates")

    # ── version ──────────────────────────────────────────────────────
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    mind_path = Path(args.path).expanduser() if hasattr(args, "path") else \
                Path("~/.aalgoi/mind").expanduser()

    if args.command == "solve":
        _cmd_solve(args, mind_path)
    elif args.command == "status":
        _cmd_status(mind_path)
    elif args.command == "train":
        _cmd_train(args, mind_path)
    elif args.command == "benchmark":
        _cmd_benchmark(args, mind_path)
    elif args.command == "checkpoint":
        _cmd_checkpoint(args, mind_path)
    elif args.command == "rollback":
        _cmd_rollback(args, mind_path)
    elif args.command == "share":
        _cmd_share(mind_path)
    elif args.command == "receive":
        _cmd_receive(mind_path)
    elif args.command == "version":
        import aalgoi
        print(f"aalgoi {aalgoi.__version__}")
    else:
        parser.print_help()


def _cmd_solve(args, mind_path: Path) -> None:
    from aalgoi._core import Mind
    from aalgoi._data import normalize

    data = None
    if args.data:
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError:
            data = args.data
    elif args.file:
        data = normalize(args.file)

    mind = Mind(mind_path)
    result = mind.solve(args.problem, data)

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
        print(result.explain())

    if args.code and result.code:
        print()
        print(result.code)


def _cmd_status(mind_path: Path) -> None:
    from aalgoi._core import Mind
    mind = Mind(mind_path)
    print(mind.status())


def _cmd_train(args, mind_path: Path) -> None:
    from aalgoi._core import Mind
    mind = Mind(mind_path)
    print(f"Training for {args.epochs} epochs...")
    stats = mind.train(epochs=args.epochs, augmentations=args.augmentations)

    print(f"\nTrajectories: {stats['n_trajectories']}")
    print(f"Steps:        {stats['n_steps']}")
    if stats.get("phase1_loss"):
        print(f"Final loss:   {stats['phase1_loss'][-1]:.4f}")
    print("\nDone.")


def _cmd_benchmark(args, mind_path: Path) -> None:
    from aalgoi._core import Mind
    mind = Mind(mind_path)
    report = mind.benchmark(verbose=args.verbose)
    print(report)


def _cmd_checkpoint(args, mind_path: Path) -> None:
    from aalgoi._core import Mind
    mind = Mind(mind_path)

    if args.action == "save":
        path = mind.checkpoint()
        if path:
            print(f"Checkpoint saved: {path}")
        else:
            print("No checkpoint saved (safety may not be available)")
    elif args.action == "list":
        ckpt_dir = mind_path / "checkpoints"
        if ckpt_dir.exists():
            files = sorted(ckpt_dir.glob("*.pt"))
            if files:
                for f in files:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"  {f.name}  ({size_mb:.1f} MB)")
            else:
                print("No checkpoints found")
        else:
            print("No checkpoints directory")


def _cmd_rollback(args, mind_path: Path) -> None:
    from aalgoi._core import Mind
    mind = Mind(mind_path)
    result = mind.rollback(args.target)
    if result.get("success"):
        print(f"Rolled back to: {result.get('target', args.target)}")
    else:
        print(f"Rollback failed: {result.get('error', 'unknown')}", file=sys.stderr)
        sys.exit(1)


def _cmd_share(mind_path: Path) -> None:
    from aalgoi._core import Mind
    mind = Mind(mind_path)
    count = mind.share()
    print(f"{count} updates ready to share in outbox")


def _cmd_receive(mind_path: Path) -> None:
    from aalgoi._core import Mind
    mind = Mind(mind_path)
    result = mind.receive()
    print(f"Processed: {result['updates_processed']}")
    print(f"Algorithms imported: {result['algorithms_imported']}")
    if result.get("errors"):
        print(f"Errors: {result['errors']}")


if __name__ == "__main__":
    main()
