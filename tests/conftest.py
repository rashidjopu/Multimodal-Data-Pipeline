"""Shared pytest fixtures/config for the test suite."""

import sys
from pathlib import Path

# Ensure `src` is importable when pytest is invoked from anywhere within the repo.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
