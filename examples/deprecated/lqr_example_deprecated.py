from control import PriorFactor, LQRTripletFactor, LQRInitialFactor
from jax import numpy as jnp
import jaxfg
from jaxfg.core import RealVectorVariable
from jaxfg.solvers import LevenbergMarquardtSolver
from typing import List
from utils.visualization_utils import LQRVis

dim_x = 2
dim_u = 1

A = jnp.array([[1.1, 0.0], [0.1, 1.1]])   # state transition linear operator on x: Ax
B = jnp.array([[0.1], [0.0]])  # state transition linear operator on u: Bu
c = jnp.array([-1., -2.])  # state transition bias, i.e., x(t+1) = Ax(t) + Bu(t) + c

Q_inv = jnp.linalg.inv(jnp.array([[10., 0.], [0., 10.]]))  # covariance of state prior
R_inv = jnp.linalg.inv(jnp.array([[1.0]]))  # covariance of action prior

cov_dyn = jnp.array([1e-10]*dim_x)  # covariance of dynamics
X0 = jnp.array([5., 5.])  # initial state
Xg = jnp.array([10., 10.])  # goal state
ug = jnp.array([0.])  # goal action
T = 60  # horizon

# ============================ build factor graphs ==========================
state_variables = [RealVectorVariable[dim_x]() for _ in range(T)]
action_variables = [RealVectorVariable[dim_u]() for _ in range(T)]

action_state_factors: List[jaxfg.core.FactorBase] = \
    [LQRInitialFactor.make(X0,
                           action_variables[0],
                           state_variables[0], A, B, c,
                           jaxfg.noises.DiagonalGaussian.make_from_covariance(cov_dyn))
     ]

state_action_state_factors: List[jaxfg.core.FactorBase] = \
    [LQRTripletFactor.make(state_variables[i],
                           action_variables[i+1],
                           state_variables[i+1],
                           A, B, c,
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
with jaxfg.utils.stopwatch("First solve (slower because of JIT compilation)"):
    solution_assignments = graph.solve(initial_assignments)#, solver=LevenbergMarquardtSolver())
    solution_assignments.storage.block_until_ready()  # type: ignore

with jaxfg.utils.stopwatch("Solve after initial compilation"):
    solution_assignments = graph.solve(initial_assignments)#, solver=LevenbergMarquardtSolver())
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
xs1 = [X0[0]] + xs1
xs2 = [X0[1]] + xs2
us = us + [0]

ts = list(range(T+1))
LQRVis.init_plot()
LQRVis.plot_trajectory(xs1, xs2, us, T+1)




