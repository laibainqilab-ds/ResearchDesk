import sys
from pathlib import Path

# Ensure the repo root is importable as `app.*` regardless of how pytest is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
