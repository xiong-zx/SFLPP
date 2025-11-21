"""Generic utility functions for optimization models."""

import json
from pathlib import Path
from typing import Dict, Optional, Any
import gurobipy as gp


def solve_gurobi_model(
    model: gp.Model,
    log_file: Optional[Path | str] = None,
    params: Optional[Dict[str, float | str]] = None,
) -> None:
    """
    Generic solve function for a Gurobi model.

    params: dict of Gurobi parameters, e.g.
        {"OutputFlag": 1, "TimeLimit": 300, "MIPGap": 0.01}
    """
    if params:
        for k, v in params.items():
            model.setParam(k, v)
    if log_file is not None:
        model.setParam("LogFile", str(log_file))
    model.optimize()


def load_gurobi_params(path: str | Path) -> Dict[str, float | str]:
    """
    Load Gurobi parameters from a JSON file.
    Returns an empty dict if the file does not exist.
    """
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    if not isinstance(params, dict):
        raise ValueError(f"Gurobi params file {path} must contain a JSON object.")
    return params
