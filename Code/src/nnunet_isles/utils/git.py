"""Current git SHA helper for leaderboard rows."""

from __future__ import annotations

import subprocess
from pathlib import Path


def current_git_sha(repo_path: Path | None = None) -> str:
    """Return short git SHA, or 'unknown' if not a repo / git unavailable."""
    cwd = str(repo_path) if repo_path is not None else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"
