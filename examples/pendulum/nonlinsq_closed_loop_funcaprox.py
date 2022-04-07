import jax
import jaxfg
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from control import PriorFactor, TransformedPriorFactor, GeneralFactorSAS, GeneralFactorAS, BoundedRealVectorVariable
from environments.pendulum import pendulum_dynamics
from jaxfg.core import RealVectorVariable
from jax import numpy as jnp
from jax import random
from jaxfg.solvers import LevenbergMarquardtSolver
from learning.configs.default import get_config
from models.networks import MLP
from typing import List
from utils.training_utils import restore_checkpoint, create_train_state
from utils.visualization_utils import PendulumMPCVis

H = 10  # how many steps we will lookahead when planning; planning_horizon

dim_x = 2
dim_u = 1
X0 = jnp.array([jnp.pi, 0.])   # [\theta, \dot{\theta}]
Xag = jnp.array([0., 1., 0.])   # goal state [sin(\theta), cos(\theta), \dot{\theta}]
Q_inv = jnp.diag(jnp.array([100, 1., 100]))  # covariance of transformed state
R_inv = jnp.diag(jnp.array([50.]))  # covariance of action
cov_dyn = jnp.array([[1e-6, 0.], [0., 1e-6]])  # covariance of state transition; small value means deterministic
T = 100  # horizon
max_u = 5
min_u = -5
key = random.PRNGKey(42)
state_variables = [RealVectorVariable[dim_x]() for _ in range(H)]
action_variables = [BoundedRealVectorVariable(min_u, max_u)[dim_u]() for _ in range(H)]

# params = jnp.load('data/pendulum_determinstic_NN_state.npy')
config = get_config()
workdir = 'data/checkpoint_62500'
rng = jax.random.PRNGKey(0)
ckpt = create_train_state(rng, config)
ckpt = restore_checkpoint(ckpt, workdir)
params = ckpt.params

def pendulum_dynamics_learned(state, action):
    if len(state.shape) == 1:
        assert len(state.shape) == len(action.shape)
        state = jnp.reshape(state, (1, -1))
        action = jnp.reshape(action, (1, -1))
    return MLP([128, 64, 20, 2]).apply({'params': params}, jnp.concatenate([state, action], 1)).reshape(-1)

action_state_factors: List[jaxfg.core.FactorBase] = \
    [GeneralFactorAS.make(X0,
                          action_variables[0],
                          state_variables[0],
                          pendulum_dynamics_learned,
                          jaxfg.noises.Gaussian.make_from_covariance(cov_dyn))]

state_action_state_factors: List[jaxfg.core.FactorBase] = \
    [GeneralFactorSAS.make(state_variables[i],
                           action_variables[i+1],
                           state_variables[i+1],
                           pendulum_dynamics_learned,
                           jaxfg.noises.Gaussian.make_from_covariance(cov_dyn))
     for i in range(H-1)]

state_prior_factors: List[jaxfg.core.FactorBase] = \
    [TransformedPriorFactor.make(state_variables[i],
                                 Xag,
                                 jaxfg.noises.Gaussian.make_from_covariance(Q_inv))
     for i in range(H)]

action_prior_factors: List[jaxfg.core.FactorBase] = \
    [PriorFactor.make(action_variables[i],
                      jnp.zeros(dim_u),
                      jaxfg.noises.Gaussian.make_from_covariance(R_inv))
     for i in range(H)]


factors: List[jaxfg.core.FactorBase] = action_state_factors \
                                       + state_action_state_factors \
                                       + state_prior_factors \
                                       + action_prior_factors

state_action_variables = state_variables + action_variables

# import ipdb; ipdb.set_trace()
graph = jaxfg.core.StackedFactorGraph.make(factors)
initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(state_action_variables)
print("Initial assignments:")
print(initial_assignments)

# Solve. Note that the first call to solve() will be much slower than subsequent calls.
with jaxfg.utils.stopwatch("First solve (slower because of JIT compilation)"):
    solution_assignments = graph.solve(initial_assignments, solver=LevenbergMarquardtSolver())
    solution_assignments.storage.block_until_ready()  # type: ignore

states_observed = [X0]
actions_taken = []

for t in range(T):
    # import ipdb; ipdb.set_trace()
    graph.factor_stacks[0].factor.initial_state[:] = states_observed[-1]
    with jaxfg.utils.stopwatch("Solve after initial compilation"):
        solution_assignments = graph.solve(initial_assignments, solver=LevenbergMarquardtSolver(max_iterations=1000))
        solution_assignments.storage.block_until_ready()  # type: ignore
    action = solution_assignments.get_value(action_variables[0])  # take the first action
    actions_taken.append(action)
    # import ipdb; ipdb.set_trace()
    new_key, subkey = random.split(key)
    del key
    next_state = pendulum_dynamics(states_observed[-1], action) \
                 + jax.random.multivariate_normal(subkey, jnp.zeros(dim_x), cov_dyn)  # run in the true environment
    states_observed.append(next_state)
    del subkey
    key = new_key

# ======================== below is for visualization ==============================
true_xs1 = [_[0] for _ in states_observed]
true_xs2 = [_[1] for _ in states_observed]
us = actions_taken + [0.]

PendulumMPCVis.init_plot()
PendulumMPCVis.plot_trajectory(true_xs1, true_xs2, us, T+1,  save_path='assets/pendulum_MPC_funcapprox_watson20.png')
