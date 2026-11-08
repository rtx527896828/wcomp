"""Code version string shared by every results writer (moved verbatim from scripts/run_baselines.py)."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code_hash() -> str:
    """Short sha256 over the framework sources (wcomp/**/*.py, scripts/*.py) and uv.lock: changes whenever they do."""
    h = hashlib.sha256()
    files = list(ROOT.glob("wcomp/**/*.py")) + list(ROOT.glob("scripts/*.py")) + [ROOT / "uv.lock"]
    for p in sorted(p for p in files if p.exists()):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:10]


def code_version() -> str:
    """'<git short hash>[+dirty]/<source hash>'; the source hash identifies uncommitted code too."""
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True)
        git = rev.stdout.strip() + ("+dirty" if status.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        git = "unknown"
    return f"{git}/{code_hash()}"
