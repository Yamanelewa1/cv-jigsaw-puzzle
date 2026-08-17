#!/usr/bin/env python
"""Minimal test runner, so the suite also works without ``pytest`` installed.

Usage::

    python tests/run_tests.py            # everything
    python tests/run_tests.py assembly   # only files matching "assembly"

With pytest available, ``python -m pytest tests`` does the same thing.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load(path):
    name = "t_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv):
    pattern = argv[0] if argv else ""
    files = sorted(f for f in os.listdir(HERE)
                   if f.startswith("test_") and f.endswith(".py")
                   and pattern in f)
    passed = failed = 0
    failures = []
    t0 = time.perf_counter()

    for fname in files:
        module = load(os.path.join(HERE, fname))
        names = [n for n in dir(module) if n.startswith("test_")]
        print(f"\n{fname}  ({len(names)} tests)")
        for name in names:
            fn = getattr(module, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print(f"  .  {name}")
            except Exception:                        # noqa: BLE001
                failed += 1
                failures.append((fname, name, traceback.format_exc()))
                print(f"  F  {name}")

    print("\n" + "=" * 70)
    for fname, name, tb in failures:
        print(f"FAILED {fname}::{name}\n{tb}")
    print(f"{passed} passed, {failed} failed in "
          f"{time.perf_counter() - t0:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
