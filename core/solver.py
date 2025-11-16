"""Generic utility functions for optimization models."""

from typing import Dict, Optional
import gurobipy as gp


def solve_gurobi_model(
    model: gp.Model,
    params: Optional[Dict[str, float]] = None,
) -> None:
    """
    Generic solve function for a Gurobi model.

    params: dict of Gurobi parameters, e.g.
        {"OutputFlag": 1, "TimeLimit": 300, "MIPGap": 0.01}
    """
    if params:
        for k, v in params.items():
            model.setParam(k, v)
    model.optimize()
