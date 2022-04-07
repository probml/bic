import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from control import PriorFactor, TransformedPriorFactor, GeneralFactorSAS, GeneralFactorAS, BoundedRealVectorVariable
import jax
from jax import random
from jax import numpy as jnp
import jaxfg
from jaxfg.solvers import LevenbergMarquardtSolver
from jaxfg.core import RealVectorVariable
from environments.pendulum import pendulum_dynamics
import matplotlib.pyplot as plt
from typing import List
from utils.visualization_utils import PendulumMPCVis
from utils.data_utils import transform_trajs_to_training_data, make_train_test_split

H = 50  # how many steps we will lookahead when planning; planning_horizon

dim_x = 2
dim_u = 1
X0 = jnp.array([jnp.pi, 0.])   # [\theta, \dot{\theta}]
Xag = jnp.array([0., 1., 0.])   # goal state [sin(\theta), cos(\theta), \dot{\theta}]
Q_inv = jnp.diag(jnp.array([100, 1., 100]))  # covariance of transformed state
R_inv = jnp.diag(jnp.array([50.]))  # covariance of action
cov_dyn = jnp.array([[0.05, 0.], [0., 1e-6]])  # covariance of state transition; small value means deterministic
T = 100  # horizon
max_u = 4
min_u = -4

key = random.PRNGKey(42)


state_variables = [RealVectorVariable[dim_x]() for _ in range(H)]
action_variables = [BoundedRealVectorVariable(min_u, max_u)[dim_u]() for _ in range(H)]

action_state_factors: List[jaxfg.core.FactorBase] = \
    [GeneralFactorAS.make(X0,
                          action_variables[0],
                          state_variables[0],
                          pendulum_dynamics,
                          jaxfg.noises.Gaussian.make_from_covariance(cov_dyn))]

state_action_state_factors: List[jaxfg.core.FactorBase] = \
    [GeneralFactorSAS.make(state_variables[i],
                           action_variables[i+1],
                           state_variables[i+1],
                           pendulum_dynamics,
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
# import ipdb; ipdb.set_trace()
# graph.factor_stacks[0].factor.initial_state[:] = X0
# import ipdb; ipdb.set_trace()
initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(state_action_variables)
print("Initial assignments:")
print(initial_assignments)

# t = 0
# initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(state_action_variables)
# initial_assignments.set_value(state_variables[t], X0)

# Solve. Note that the first call to solve() will be much slower than subsequent calls.
with jaxfg.utils.stopwatch("First solve (slower because of JIT compilation)"):
    solution_assignments = graph.solve(initial_assignments, solver=LevenbergMarquardtSolver())
    solution_assignments.storage.block_until_ready()  # type: ignore

n_trajs = 100
trajs = []

for i in range(n_trajs):
    new_key, subkey = random.split(key)
    X0 = jax.random.uniform(subkey, shape=X0.shape, minval=-3*jnp.pi, maxval=3*jnp.pi)
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
    # import ipdb; ipdb.set_trace()
    Xs = jnp.stack(states_observed, 0)
    Us = jnp.stack(actions_taken + [jnp.zeros(dim_u)], 0)
    traj = jnp.concatenate([Xs, Us], 1)
    trajs.append(traj)

trajs = jnp.stack(trajs, 0)
Xs, Ys = transform_trajs_to_training_data(trajs, dim_x, dim_u)

train_X, train_Y, test_X, test_Y = make_train_test_split(Xs, Ys, 0.2)

dataset = {'train_x': train_X, 'train_y': train_Y, 'test_x': test_X, 'test_y': test_Y}

jnp.save('pendulum_determinstic_dataset', dataset)



    # ======================== below is for visualization ==============================


# PendulumMPCVis.init_plot()
# PendulumMPCVis.plot_trajectory(true_xs1, true_xs2, us, T+1) # , save_path='assets/pendulum_MPC_watson20.png')
