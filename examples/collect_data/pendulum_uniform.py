import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

import jax
import jax.numpy as jnp

from environments.pendulum import pendulum_dynamics
from jax import random
from utils.data_utils import make_train_test_split

if __name__ == '__main__':
    key = random.PRNGKey(42)
    newkey, subkey = random.split(key)

    x0 = jnp.array([jnp.pi, 0.])
    u0 = jnp.array([0])

    n_data = 200  # 10 thousands

    x_max, x_min = 2 * jnp.pi, -2 * jnp.pi
    u_max, u_min = 5, -5

    states = jax.random.uniform(subkey, shape=(n_data, 2), minval=x_min, maxval=x_max)
    actions = jax.random.uniform(subkey, shape=(n_data, 1), minval=u_min, maxval=u_max)

    Xs = jnp.concatenate([states, actions], 1)

    Ys = pendulum_dynamics(states, actions)
    # import ipdb; ipdb.set_trace()

    train_X, train_Y, test_X, test_Y = make_train_test_split(Xs, Ys, 0.2)

    dataset = {'train_x': train_X, 'train_y': train_Y, 'test_x': test_X, 'test_y': test_Y}

    jnp.save('data/pendulum_determinstic_uniform_dataset-%d' % n_data, dataset)








