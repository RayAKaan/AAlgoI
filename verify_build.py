#!/usr/bin/env python3
"""
Verify that the package builds and installs correctly.
"""

import subprocess
import sys


def run_command(cmd, description):
    """Run a shell command and return success status."""
    print(f"  {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  OK - {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED - {description}")
        print(e.stderr)
        return False


def main():
    print("=" * 60)
    print("AAlgoI Build Verification")
    print("=" * 60)

    steps = [
        ("python -m pip install --upgrade pip", "Upgrade pip"),
        ("python -m pip install -r requirements.txt", "Install dependencies"),
        ("python -m pip install build", "Install build tool"),
        ("python -m build", "Build package"),
        ("python -m pip install -e .", "Install in editable mode"),
        ("python -c \"from pipeline import UniversalSolver; s=UniversalSolver(); print('Import OK')\"", "Test import"),
        ("python demo.py", "Run demo"),
        ("python -m pytest tests/test_full_integration.py -v", "Run test suite"),
    ]

    results = []
    for cmd, desc in steps:
        results.append(run_command(cmd, desc))

    print("=" * 60)
    if all(results):
        print("All checks passed! Package is ready to ship.")
        sys.exit(0)
    else:
        print("Some checks failed. Please review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
