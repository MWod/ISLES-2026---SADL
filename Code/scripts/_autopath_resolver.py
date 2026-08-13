"""Register Hydra's `${autopath:key}` resolver, backed by the paths module.

Import + call `register_autopath_resolver()` BEFORE Hydra parses configs.
"""

from __future__ import annotations


def register_autopath_resolver() -> None:
    import paths
    from omegaconf import OmegaConf

    if OmegaConf.has_resolver("autopath"):
        return

    def _resolve(key: str) -> str:
        value = getattr(paths, key, None)
        if value is None:
            raise KeyError(f"autopath: paths module has no attribute '{key}'")
        return str(value)

    OmegaConf.register_new_resolver("autopath", _resolve)
