import jax.numpy as jnp
import jax

@jax.jit
def pendulum_dynamics(state: jnp.ndarray,
                      action: jnp.ndarray) -> jnp.ndarray:
    dt = 0.05
    m = 1.0
    l = 1.0
    d = 1e-2  # damping
    g = 9.80665
    u_mx = 5.

    batch_mode = len(state.shape) ==2

    if not batch_mode:
        assert len(state.shape) == len(action.shape)
        state = jnp.reshape(state, (1, -1))
        action = jnp.reshape(action, (1, -1))

    x = state
    u = action
    u = jnp.clip(u, -u_mx, u_mx)
    th_dot_dot = -3.0 * g / (2 * l) * jnp.sin(x[:, 0] + jnp.pi) - d * x[:, 1]
    th_dot_dot += 3.0 / (m * l ** 2) * u.squeeze()
    x_dot = x[:, 1] + th_dot_dot * dt
    x_pos = x[:, 0] + x_dot * dt
    x2 = jnp.vstack((x_pos, x_dot)).T
    if not batch_mode:
        return x2.reshape(-1)
    else:
        return x2


if __name__ == "__main__":
    # print_yay(1, 2)
    x = jnp.array([1, 1.])
    u = jnp.array([1.])
    print(pendulum_dynamics(x, u), type(pendulum_dynamics(x, u)))
