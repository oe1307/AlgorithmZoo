from solver.branch_and_bound import BranchAndBound
from solver.simplex_method import SimplexMethod
from solver.interior_point_method import InteriorPointMethod
from solver._pulp import Pulp


def get_solver(name):
    if name == "branch":
        solver = BranchAndBound
    elif name == "simplex":
        solver = SimplexMethod
    elif name == "pulp":
        solver = Pulp
    else:
        raise NotImplementedError(name)
    return solver
