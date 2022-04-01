from typing import NamedTuple, Tuple

import jax_dataclasses as jdc
from jax import numpy as jnp
from overrides import overrides

from jaxfg import noises
from jaxfg.core._factor_base import FactorBase


PriorValueTuple = Tuple[jnp.ndarray]


@jdc.pytree_dataclass
class PriorFactor(FactorBase[PriorValueTuple]):
    """Factor for defining a fixed prior on a frame.
    Residuals are computed as `(variable.inverse() @ mu).log()`.
    """

    mu: jnp.ndarray

    @staticmethod
    def make(
        variable: jnp.ndarray,
        mu: jnp.ndarray,
        noise_model: noises.NoiseModelBase,
    ) -> "PriorFactor":
        return PriorFactor(
            variables=(variable,),
            mu=mu,
            noise_model=noise_model,
        )

    @overrides
    def compute_residual_vector(self, variable_values: PriorValueTuple) -> jnp.ndarray:

        T: jnp.ndarray
        (T,) = variable_values

        # Equivalent to: return (variable_value.inverse() @ self.mu).log()
        # FIXME(CW): does the sign of the residual matter?
        return T - self.mu


@jdc.pytree_dataclass
class TransformedPriorFactor(FactorBase[PriorValueTuple]):
    """Factor for defining a fixed prior on a frame.
    Residuals are computed as `(variable.inverse() @ mu).log()`.
    """

    mu: jnp.ndarray

    @staticmethod
    def make(
        variable: jnp.ndarray,
        mu: jnp.ndarray,
        noise_model: noises.NoiseModelBase,
    ) -> "TransformedPriorFactor":
        return TransformedPriorFactor(
            variables=(variable,),
            mu=mu,
            noise_model=noise_model,
        )

    @overrides
    def compute_residual_vector(self, variable_values: PriorValueTuple) -> jnp.ndarray:

        T: jnp.ndarray
        (T,) = variable_values
        T_transformed = jnp.concatenate([jnp.sin(T[0:1]), jnp.cos(T[0:1]), T[1:2]])

        # Equivalent to: return (variable_value.inverse() @ self.mu).log()
        return T_transformed - self.mu


class LQRTripletTuple(NamedTuple):
    prev_state: jnp.ndarray
    action: jnp.ndarray
    next_state: jnp.ndarray


class LQRInitialTuple(NamedTuple):
    action: jnp.ndarray
    next_state: jnp.ndarray


@jdc.pytree_dataclass
class LQRInitialFactor(FactorBase[LQRInitialTuple]):

    initial_state: jnp.ndarray
    A: jnp.ndarray
    B: jnp.ndarray
    c: jnp.ndarray

    @staticmethod
    def make(
        prev_state: jnp.ndarray,
        action: jnp.ndarray,
        next_state: jnp.ndarray,
        A: jnp.ndarray,
        B: jnp.ndarray,
        c: jnp.ndarray,
        noise_model: noises.NoiseModelBase,
    ) -> "LQRInitialFactor":
        return LQRInitialFactor(
            variables=(
                action,
                next_state
            ),
            A=A,
            B=B,
            c=c,
            initial_state=prev_state,
            noise_model=noise_model,
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: LQRInitialTuple
    ) -> jnp.ndarray:
        prev_state = self.initial_state
        next_state = variable_values.next_state
        action = variable_values.action
        return jnp.dot(self.A, prev_state) + jnp.dot(self.B, action) + self.c - next_state


@jdc.pytree_dataclass
class LQRTripletFactor(FactorBase[LQRTripletTuple]):

    A: jnp.ndarray
    B: jnp.ndarray
    c: jnp.ndarray

    @staticmethod
    def make(
        prev_state: jnp.ndarray,
        action: jnp.ndarray,
        next_state: jnp.ndarray,
        A: jnp.ndarray,
        B: jnp.ndarray,
        c: jnp.ndarray,
        noise_model: noises.NoiseModelBase,
    ) -> "LQRTripletFactor":
        assert type(prev_state) is type(next_state)
        return LQRTripletFactor(
            variables=(
                prev_state,
                action,
                next_state
            ),
            A=A,
            B=B,
            c=c,
            noise_model=noise_model,
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: LQRTripletTuple
    ) -> jnp.ndarray:
        prev_state = variable_values.prev_state
        next_state = variable_values.next_state
        action = variable_values.action
        return  jnp.dot(self.A, prev_state) + jnp.dot(self.B, action) + self.c - next_state





