"""Decorator-based registries for swappable components.

Each component category (trainers, networks, losses, augmentation transforms,
harmonizers, metadata-conditioning blocks, splits, planners, preprocessors)
has its own Registry instance. Components register themselves via decorator
and are instantiated by name from Hydra configs.
"""

from collections.abc import Callable
from typing import Any


class Registry:
    """A named registry mapping string keys to classes."""

    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, type] = {}

    def register(self, key: str) -> Callable:
        def decorator(cls: type) -> type:
            if key in self._registry:
                raise ValueError(
                    f"Registry '{self.name}': key '{key}' already registered to "
                    f"{self._registry[key].__name__}. Cannot re-register {cls.__name__}."
                )
            self._registry[key] = cls
            return cls

        return decorator

    def build(self, key: str, **kwargs: Any) -> Any:
        if key not in self._registry:
            raise KeyError(
                f"Registry '{self.name}': key '{key}' not found. "
                f"Available: [{', '.join(sorted(self._registry))}]"
            )
        return self._registry[key](**kwargs)

    def get(self, key: str) -> type:
        if key not in self._registry:
            raise KeyError(
                f"Registry '{self.name}': key '{key}' not found. "
                f"Available: [{', '.join(sorted(self._registry))}]"
            )
        return self._registry[key]

    def keys(self) -> list[str]:
        return list(self._registry)

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name='{self.name}', keys={self.keys()})"


TRAINER_REGISTRY = Registry("trainer")
NETWORK_REGISTRY = Registry("network")
LOSS_REGISTRY = Registry("loss")
AUGMENTATION_REGISTRY = Registry("augmentation")
HARMONIZER_REGISTRY = Registry("harmonizer")
META_COND_REGISTRY = Registry("metadata_conditioning")
SPLIT_REGISTRY = Registry("split_strategy")
PLANNER_REGISTRY = Registry("planner")
PREPROCESSOR_REGISTRY = Registry("preprocessor")
