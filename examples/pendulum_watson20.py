import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from control import PriorFactor, TransformedPriorFactor, GeneralFactorSAS, GeneralFactorAS
from jax import numpy as jnp
import jaxfg
from jaxfg.solvers import LevenbergMarquardtSolver
from jaxfg.core import RealVectorVariable
from environments.pendulum import pendulum_dynamics
import matplotlib.pyplot as plt
from typing import List
from utils.visualization_utils import PendulumVis


dim_x = 2
dim_u = 1
X0 = jnp.array([jnp.pi/20, 0.0])   # [\theta, \dot{\theta}]
Xag = jnp.array([0., 1., 0.])   # goal state [sin(\theta), cos(\theta), \dot{\theta}]
Q_inv = jnp.diag(jnp.array([100, 1e-3, 100]))  # covariance of transformed state
R_inv = jnp.diag(jnp.array([50]))  # covariance of action
cov_dyn = jnp.array([[1e-5, 0.], [0., 1e-5]])  # covariance of state transition; small value means deterministic
T = 20  # horizon

state_variables = [RealVectorVariable[dim_x]() for _ in range(T)]
action_variables = [RealVectorVariable[dim_u]() for _ in range(T)]

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
     for i in range(T-1)]

state_prior_factors: List[jaxfg.core.FactorBase] = \
    [TransformedPriorFactor.make(state_variables[i],
                                 Xag,
                                 jaxfg.noises.Gaussian.make_from_covariance(Q_inv))
     for i in range(T)]

action_prior_factors: List[jaxfg.core.FactorBase] = \
    [PriorFactor.make(action_variables[i],
                      jnp.zeros(dim_u),
                      jaxfg.noises.Gaussian.make_from_covariance(R_inv))
     for i in range(T)]

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

with jaxfg.utils.stopwatch("Solve after initial compilation"):
    solution_assignments = graph.solve(initial_assignments, solver=LevenbergMarquardtSolver())
    solution_assignments.storage.block_until_ready()  # type: ignore

# Print all solved variable values.
print("Solutions (jaxfg.core.VariableAssignments):")
print(solution_assignments)
print()
# import ipdb;ipdb.set_trace()


# ======================== below is for visualization ==============================
us = [solution_assignments.get_value(action_variables[i]) for i in range(T)]

xs1 = [solution_assignments.get_value(state_variables[i])[0] for i in range(T)]
xs2 = [solution_assignments.get_value(state_variables[i])[1] for i in range(T)]
xs = [X0] + [solution_assignments.get_value(state_variables[i]) for i in range(T)]

xs1 = [X0[0]] + xs1  # append the initial location at the beginning of the list
xs2 = [X0[1]] + xs2
us = us + [0]  # append the last action at the end of the list

# below is for get the trajectories under the true dynamics.
true_xs = [X0]  # append the first location
for i in range(T):
    x_prev = true_xs[-1]
    next_x = pendulum_dynamics(x_prev, us[i])
    true_xs.append(next_x)

true_xs1 = [_[0] for _ in true_xs]
true_xs2 = [_[1] for _ in true_xs]
ts = list(range(T+1))

PendulumVis.init_plot()
PendulumVis.plot_trajectory(xs1, xs2, true_xs1, true_xs2, us, T+1, save_path='assets/pendulum_watson20.png')

