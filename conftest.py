# conftest.py — repo-root pytest fixture: put the project root on sys.path so tests can
# `import reward`, `import experiments`, `import env`, ... without per-file sys.path hacks.
"""Pytest root conftest: ensure the project root is importable in all test modules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
