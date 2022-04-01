from typing import NamedTuple, Tuple, Callable

import jax_dataclasses as jdc
from jax import numpy as jnp
from overrides import overrides

from jaxfg import noises
from jaxfg.core._factor_base import FactorBase


class StateActionStateTriplet(NamedTuple):
    prev_state: jnp.ndarray
    action: jnp.ndarray
    next_state: jnp.ndarray


class ActionStateTuple(NamedTuple):
    action: jnp.ndarray
    next_state: jnp.ndarray


class GeneralFactorSAS:

    @staticmethod
    def make(
        prev_state: jnp.ndarray,
        action: jnp.ndarray,
        next_state: jnp.ndarray,
        transit_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        noise_model: noises.NoiseModelBase,
    ):
        @jdc.pytree_dataclass
        class _Temp(FactorBase[StateActionStateTriplet]):
            def transition_function(
                    self, s0: jnp.ndarray, a0: jnp.ndarray
            ) -> jnp.ndarray:
                return transit_function(s0, a0)

            @overrides
            def compute_residual_vector(
                    self, variable_values: StateActionStateTriplet
            ) -> jnp.ndarray:
                s0 = variable_values.prev_state
                s1 = variable_values.next_state
                a0 = variable_values.action
                return transit_function(s0, a0) - s1

        factor = _Temp(variables=(prev_state, action, next_state,), noise_model=noise_model,)
        return factor


class GeneralFactorAS:

    @staticmethod
    def make(
            prev_state: jnp.ndarray,
            action: jnp.ndarray,
            next_state: jnp.ndarray,
            transit_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
            noise_model: noises.NoiseModelBase,
    ) -> FactorBase[ActionStateTuple]:
        @jdc.pytree_dataclass
        class _Temp(FactorBase[ActionStateTuple]):
            initial_state: jnp.ndarray

            def transition_function(
                    self, s0: jnp.ndarray, a0: jnp.ndarray
            ) -> jnp.ndarray:
                return transit_function(s0, a0)

            @overrides
            def compute_residual_vector(
                    self, variable_values: ActionStateTuple
            ) -> jnp.ndarray:
                s1 = variable_values.next_state
                a0 = variable_values.action
                return transit_function(self.initial_state, a0) - s1

        factor = _Temp(variables=(action, next_state,), noise_model=noise_model, initial_state=prev_state)
        return factor
