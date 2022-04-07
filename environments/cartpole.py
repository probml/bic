import jax
import jax.numpy as jnp


@jax.jit
def cartpole_dynamics(state, action):

    g = 9.81
    Mc = 0.37
    Mp = 0.127
    Mt = Mc + Mp
    l = 0.3365
    fs_hz = 250.0
    dt = 1 / fs_hz
    u_mx = 5.0

    if len(state.shape) == 1:
        assert len(state.shape) == len(action.shape)
        state = jnp.reshape(state, (1, -1))
        action = jnp.reshape(action, (1, -1))

    x = state
    u = action

    _u = jnp.clip(u, -u_mx, u_mx).squeeze()

    th = x[:, 1]
    dth2 = jnp.power(x[:, 3], 2)
    sth = jnp.sin(th)
    cth = jnp.cos(th)

    _num = -Mp * l * sth * cth * dth2 + Mt * g * sth - _u * cth
    _denom = l * ((4.0 / 3.0) * Mt - Mp * cth ** 2)
    th_acc = _num / _denom
    x_acc = (Mp * l * sth * dth2 - Mp * l * th_acc * cth + _u) / Mt

    y1 = x[:, 0] + dt * x[:, 2]
    y2 = x[:, 1] + dt * x[:, 3]
    y3 = x[:, 2] + dt * x_acc
    y4 = x[:, 3] + dt * th_acc

    y = jnp.vstack((y1, y2, y3, y4)).T
    return y
