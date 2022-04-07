import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from control import PriorFactor, GeneralFactorSAS, GeneralFactorAS
from environments.lqr_simple import LQREnv
from jax import numpy as jnp
import jaxfg
from jaxfg.core import RealVectorVariable
from jaxfg.solvers import LevenbergMarquardtSolver
from typing import List
from utils.visualization_utils import LQRVisGTSAM


# define the configurations of the environment
dim_x = 1
dim_u = 1

# The following transition function is wrapped as in LQREnv.lqr_simple_gtsm
# A = jnp.array([[1.03]]) # slightly unstable system :)
# B = jnp.array([[0.03]])
Q_inv = jnp.linalg.inv(jnp.array([[0.21]]))  # covariance of state prior
R_inv = jnp.linalg.inv(jnp.array([[0.05]]))  # covariance of action prior

X0 = jnp.array([-10.])  # initial state
Xg = jnp.array([0.])  # goal state
T = 100
cov_dyn = jnp.array([1e-10])

# ============================ build factor graphs ==========================
state_variables = [RealVectorVariable[dim_x]() for _ in range(T)]
action_variables = [RealVectorVariable[dim_u]() for _ in range(T)]

action_state_factors: List[jaxfg.core.FactorBase] = \
    [GeneralFactorAS.make(X0,
                          action_variables[0],
                          state_variables[0],
                          LQREnv.lqr_simple_gtsam,  # define the transition function
                          jaxfg.noises.DiagonalGaussian.make_from_covariance(cov_dyn))
     ]

state_action_state_factors: List[jaxfg.core.FactorBase] = \
    [GeneralFactorSAS.make(state_variables[i],
                           action_variables[i+1],
                           state_variables[i+1],
                           LQREnv.lqr_simple_gtsam,  # define the transition function
                           jaxfg.noises.DiagonalGaussian.make_from_covariance(cov_dyn))
     for i in range(T-1)]

state_prior_factors: List[jaxfg.core.FactorBase] = \
    [PriorFactor.make(state_variables[i],
                      Xg,
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

graph = jaxfg.core.StackedFactorGraph.make(factors)
initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(state_action_variables)
print("Initial assignments:")
print(initial_assignments)

# Solve. Note that the first call to solve() will be much slower than subsequent calls.
# The optimizer is by default using Gauss-Newton.
with jaxfg.utils.stopwatch("First solve (slower because of JIT compilation)"):
    solution_assignments = graph.solve(initial_assignments)  #, solver=LevenbergMarquardtSolver())
    solution_assignments.storage.block_until_ready()  # type: ignore

with jaxfg.utils.stopwatch("Solve after initial compilation"):
    solution_assignments = graph.solve(initial_assignments)  #, solver=LevenbergMarquardtSolver())
    solution_assignments.storage.block_until_ready()  # type: ignore

# Print all solved variable values.
print("Solutions (jaxfg.core.VariableAssignments):")
print(solution_assignments)
print()
# import ipdb;ipdb.set_trace()

# ======================== below is for visualization ==============================
us = [solution_assignments.get_value(action_variables[i]) for i in range(T)]
us = us + [0]
xs = [solution_assignments.get_value(state_variables[i]) for i in range(T)]
xs = [X0] + xs  # append the initial location at the beginning

# computing the true trajectory
true_xs = [X0]
for i in range(T):
    prev_x = true_xs[-1]
    u = us[i]
    next_x = LQREnv.lqr_simple_gtsam(prev_x, u)
    true_xs.append(next_x)


LQRVisGTSAM.init_plot()
LQRVisGTSAM.plot_trajectory(xs, true_xs, us, T+1, save_path='assets/lqr_gtsam.png')






