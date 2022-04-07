# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default Hyperparameter configuration."""
import flax.linen as nn
import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.learning_rate = 0.01
  config.momentum = 0.9
  config.batch_size = 128
  config.num_epochs = 500
  config.data_path = 'data/pendulum_determinstic_uniform_dataset-20000.npy'
  config.input_dim = 3
  config.activation = nn.swish
  config.fc_dims = [128, 64, 20, 2]
  return config
