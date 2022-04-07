import jax.numpy as jnp
import random


def transform_traj_to_input_prediction_format(traj, dim_x, dim_u):
    """
    :param traj:  T * (dim_x + dim_u)
    :return:
    """
    xs = []
    ys = []
    T = traj.shape[0]
    assert traj.shape[1] == dim_u + dim_x
    for t in range(T-1):
        x = traj[t]   # dim_x + dim_u
        y = traj[t+1][:dim_x]
        xs.append(x)
        ys.append(y)
    return jnp.stack(xs, 0), jnp.stack(ys, 0)


def transform_trajs_to_training_data(trajs, dim_x, dim_u):
    """
    :param trajs:  n_trajs * T * (dim_x + dim_u)
    :param dim_x:
    :param dim_u:
    :return:
    """
    n_trajs, n_steps, _ = trajs.shape

    Xs = []  # store state action
    Ys = []  # store the next state

    for t in range(n_trajs):
        traj = trajs[t]
        xs, ys = transform_traj_to_input_prediction_format(traj, dim_x, dim_u)
        Xs.append(xs)
        Ys.append(ys)
    return jnp.concatenate(Xs, 0), jnp.concatenate(Ys, 0)


def make_train_test_split(Xs, Ys, test_ratio=0.2):
    n_data = Xs.shape[0]
    indices = list(range(n_data))

    n_test = int(n_data * test_ratio)
    n_train = n_data - n_test

    random.shuffle(indices)

    train_indices = jnp.array(indices[:n_train])
    test_indices = jnp.array(indices[n_train:])

    train_X = Xs[train_indices]
    train_Y = Ys[train_indices]

    test_X = Xs[test_indices]
    test_Y = Ys[test_indices]

    return train_X, train_Y, test_X, test_Y




