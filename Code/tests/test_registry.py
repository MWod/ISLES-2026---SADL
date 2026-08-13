"""Smoke tests for the registry."""

from __future__ import annotations

import pytest
from nnunet_isles.registry import Registry


def test_register_and_build_roundtrip():
    reg = Registry("test")

    @reg.register("alpha")
    class Alpha:
        def __init__(self, value: int = 0):
            self.value = value

    inst = reg.build("alpha", value=42)
    assert isinstance(inst, Alpha)
    assert inst.value == 42


def test_double_register_raises():
    reg = Registry("test")

    @reg.register("alpha")
    class _Alpha:
        pass

    with pytest.raises(ValueError):

        @reg.register("alpha")
        class _Alpha2:
            pass


def test_unknown_key_raises():
    reg = Registry("test")
    with pytest.raises(KeyError):
        reg.build("ghost")
