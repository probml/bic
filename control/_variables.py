import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import functools
import jaxfg.hints as hints
from jaxfg.core import VariableBase
import jax.numpy as jnp
from overrides import EnforceOverrides, final, overrides
from typing import Type, TypeVar, Mapping

VariableType = TypeVar("VariableType", bound="VariableBase")
VariableValueType = TypeVar("VariableValueType", bound=hints.VariableValue)


class _BoundedRealVectorVariableTemplate:
    """Usage: `RealVectorVariable[N]`, where `N` is an integer dimension."""

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

   # @classmethod
    @functools.lru_cache(maxsize=None)
    def __getitem__(self, dim: int) -> Type[VariableBase]:
        assert isinstance(dim, int)

        class _BoundedRealVectorVariable(VariableBase[hints.Array]):

            @classmethod
            @overrides
            @final
            def get_default_value(cls) -> hints.Array:
                return jnp.zeros(dim)

            @classmethod
            @overrides
            @final
            def manifold_retract(
                    cls, x: VariableValueType, local_delta: hints.LocalVariableValue
            ) -> VariableValueType:
                r"""Retract local delta to manifold.
                Typically written as `x $\oplus$ local_delta` or `x $\boxplus$ local_delta`.
                Args:
                    x: Absolute parameter to update.
                    local_delta: Delta value in local parameterizaton.
                Returns:
                    Updated parameterization.
                """
                return cls.unflatten(jnp.clip(cls.flatten(x) + local_delta, self.min_val, self.max_val))

        return _BoundedRealVectorVariable


BoundedRealVectorVariable: Mapping[int, Type[VariableBase[hints.Array]]]
BoundedRealVectorVariable = _BoundedRealVectorVariableTemplate
