from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
from flax.training import checkpoints
from models.networks import MLP
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
# import tensorflow_datasets as tfds


@jax.jit
def apply_model(state, x, y):
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params):
    pred = MLP([128, 64, 20, 2], 'swish').apply({'params': params}, x)
    loss = jnp.mean(optax.l2_loss(pred, y))
    return loss, pred

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, pred), grads = grad_fn(state.params)
  return grads, loss


@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['x'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['x']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []

  for perm in perms:
    batch_x = train_ds['x'][perm, ...]
    batch_y = train_ds['y'][perm, ...]
    grads, loss = apply_model(state, batch_x, batch_y)
    state = update_model(state, grads)
    epoch_loss.append(loss)
  train_loss = np.mean(epoch_loss)
  return state, train_loss


def create_train_state(rng, config):
  """Creates initial `TrainState`."""
  mlp = MLP([128, 64, 20, 2], config.activation)
  params = mlp.init(rng, jnp.ones([1, config.input_dim]))['params']
  tx = optax.sgd(config.learning_rate, config.momentum)
  return train_state.TrainState.create(
      apply_fn=mlp.apply, params=params, tx=tx)


def get_datasets(path):
  data_load = jnp.load(path, allow_pickle=True)
  data = data_load.item()
  train_x, train_y = data['train_x'], data['train_y']
  test_x, test_y = data['test_x'], data['test_y']

  train_data = {'x': train_x, 'y': train_y}
  test_data = {'x': test_x, 'y': test_y}
  return train_data, test_data


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> train_state.TrainState:
  """Execute model training and evaluation loop.
  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    The train state (which includes the `.params`).
  """
  train_ds, test_ds = get_datasets(config.data_path)
  rng = jax.random.PRNGKey(0)

  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(dict(config))

  rng, init_rng = jax.random.split(rng)
  state= create_train_state(init_rng, config)

  for epoch in range(1, config.num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss = train_epoch(state, train_ds,
                                    config.batch_size,
                                    input_rng)
    _, test_loss = apply_model(state, test_ds['x'],
                               test_ds['y'])
    # print(MLP([128, 64, 20, 2]).apply({'params': state.params}, test_ds['x']))
    # print(test_ds['y'])
    logging.info(
        'epoch:% 3d, train_loss: %.4f, test_loss: %.4f'
        % (epoch, train_loss, test_loss))

    summary_writer.scalar('train_loss', train_loss, epoch)
    summary_writer.scalar('test_loss', test_loss, epoch)
  save_checkpoint(state, workdir)
  summary_writer.flush()
  return state


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    # state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)
