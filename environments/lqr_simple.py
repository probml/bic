import jax
import jax.numpy as jnp

class LQREnv:

    @staticmethod
    @jax.jit
    def lqr_simple_watson(s0, a0):
        A = jnp.array([[1.1, 0.0], [0.1, 1.1]])  # state transition linear operator on x: Ax
        B = jnp.array([[0.1], [0.0]])  # state transition linear operator on u: Bu
        c = jnp.array([-1., -2.])  # state transition bias, i.e., x(t+1) = Ax(t) + Bu(t) + c
        return jnp.dot(A, s0) + jnp.dot(B, a0) + c

    @staticmethod
    @jax.jit
    def lqr_simple_gtsm(s0, a0):
        A = jnp.array([[1.03]])  # slightly unstable system :)
        B = jnp.array([[0.03]])
        return jnp.dot(A, s0) + jnp.dot(B, a0)

