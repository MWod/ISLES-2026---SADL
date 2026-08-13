"""Deterministic config fingerprint for cross-experiment comparability."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_omegaconf(cfg: Any) -> str:
    """Hash a resolved OmegaConf DictConfig (or plain dict/scalar) to a 16-char SHA-256 prefix."""
    try:
        from omegaconf import OmegaConf

        # to_container raises ValueError on non-OmegaConf inputs (e.g. plain dicts).
        resolved = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    except (ImportError, AttributeError, ValueError):
        resolved = cfg
    encoded = json.dumps(resolved, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
