from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x

# import jax
# import haiku as hk
# import jax.numpy as jnp
#
# from typing import Mapping
# import numpy as np
#
# Batch = Mapping[str, jnp.ndarray]
#
#
# def pendulum_net(batch: Batch) -> jnp.ndarray:
#   x, u = batch['x'], batch['u']
#   inp = jnp.concatenate([x, u], axis=1)
#   mlp = hk.Sequential([
#       hk.Flatten(),
#       hk.Linear(20), jax.nn.relu,
#       hk.Linear(20), jax.nn.relu,
#       hk.Linear(2),
#   ])
#   return mlp(inp)
#
#
# def pendulum_net_wrap(x, u) -> jnp.ndarray:
#     batch = {'x': x, 'u': u}
#     return pendulum_net




