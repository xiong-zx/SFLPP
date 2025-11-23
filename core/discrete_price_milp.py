"""
Discrete-price MILP model for the stochastic facility location problem.

Instead of continuous prices p[i] in [p_min, p_max], we discretize prices
into L levels and use binary variables to select which level to use.
"""

from typing import Any, Dict, Tuple
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .extensive_form import ExtensiveForm


def build_discrete_price_milp_model(
    ext_form: ExtensiveForm,
    alpha: float = 0.1,
    price_levels: int = 10,
) -> Tuple[gp.Model, Dict[str, Any]]:
    """
    Build an extensive-form MILP where prices are discretized into levels.

    Args:
        ext_form: ExtensiveForm object with instance and scenarios
        alpha: Service level parameter (must serve at least (1-alpha) of demand)
        price_levels: Number of discrete price levels to use

    Returns:
        model: Gurobi model
        vars:  dict with keys 'x', 'q', 's', 'z', 'p'
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must be in [0, 1).")
    if price_levels < 2:
        raise ValueError("price_levels must be at least 2.")

    inst = ext_form.instance
    I, J, W = ext_form.I, ext_form.J, ext_form.W
    f = inst.f
    a = inst.a
    b = inst.b
    cfg = inst.config
    scenarios = ext_form.scenarios

    # Create discrete price levels
    p_min = cfg.p_min
    p_max = cfg.p_max
    price_grid = np.linspace(p_min, p_max, price_levels)
    L = list(range(price_levels))

    m = gp.Model("SCFLP_discrete_price_milp")
    m.setParam("OutputFlag", 0)

    # First-stage binaries x_j (facility location)
    x = m.addVars(J, vtype=GRB.BINARY, name="x")

    # First-stage price selection: z[i, l] = 1 if customer i has price level l
    z = m.addVars(I, L, vtype=GRB.BINARY, name="z")

    # Auxiliary continuous variable for price (for easier constraint formulation)
    p = m.addVars(I, lb=p_min, ub=p_max, vtype=GRB.CONTINUOUS, name="p")

    # Each customer must select exactly one price level
    for i in I:
        m.addConstr(
            gp.quicksum(z[i, l] for l in L) == 1,
            name=f"price_select_i{i}",
        )

    # Link p[i] to selected price level
    for i in I:
        m.addConstr(
            p[i] == gp.quicksum(price_grid[l] * z[i, l] for l in L),
            name=f"price_def_i{i}",
        )

    # Second-stage continuous q_ij^w, s_i^w
    q = m.addVars(I, J, W, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    s = m.addVars(I, W, lb=0.0, vtype=GRB.CONTINUOUS, name="s")

    # s_i^w = Σ_j q_ij^w
    for w in W:
        for i in I:
            m.addConstr(
                s[i, w] == gp.quicksum(q[i, j, w] for j in J),
                name=f"s_def_i{i}_w{w}",
            )

    # Capacity constraints: Σ_i q_ij^w ≤ u_j(w) x_j
    for w in W:
        scen = scenarios[w]
        for j in J:
            m.addConstr(
                gp.quicksum(q[i, j, w] for i in I) <= scen.u[j] * x[j],
                name=f"cap_j{j}_w{w}",
            )

    # Demand bounds: (1-alpha) h_i(p_i) ≤ s_i^w ≤ h_i(p_i)
    # where h_i(p_i) = -a[i] * p[i] + b[i]
    for w in W:
        for i in I:
            demand_at_price = -a[i] * p[i] + b[i]
            m.addConstr(
                s[i, w] <= demand_at_price,
                name=f"demand_upper_i{i}_w{w}",
            )
            m.addConstr(
                s[i, w] >= (1.0 - alpha) * demand_at_price,
                name=f"demand_lower_i{i}_w{w}",
            )

    # Objective: fixed + expected (transport/tariff - revenue)
    # Note: This is now linear because p[i] is determined by z variables
    obj = gp.LinExpr()
    obj += gp.quicksum(f[j] * x[j] for j in J)
    for w in W:
        scen = scenarios[w]
        weight = scen.weight
        for i in I:
            obj += weight * gp.quicksum(scen.bar_c[(i, j)] * q[i, j, w] for j in J)
            # Revenue term: -p[i] * s[i, w]
            # This is bilinear, but we can linearize it using the z variables
            # -p[i] * s[i, w] = -sum_l(price_grid[l] * z[i, l]) * s[i, w]
            # We need auxiliary variables for this

    # Linearize the revenue term -p[i] * s[i, w]
    # Introduce y[i, w, l] to represent z[i, l] * s[i, w]
    y = m.addVars(I, W, L, lb=0.0, vtype=GRB.CONTINUOUS, name="y")

    # Big-M for linearization (use upper bound on demand)
    M = max(b[i] for i in I) * 2  # Conservative upper bound

    for w in W:
        for i in I:
            for l in L:
                # y[i, w, l] ≤ M * z[i, l]
                m.addConstr(y[i, w, l] <= M * z[i, l], name=f"lin1_i{i}_w{w}_l{l}")
                # y[i, w, l] ≤ s[i, w]
                m.addConstr(y[i, w, l] <= s[i, w], name=f"lin2_i{i}_w{w}_l{l}")
                # y[i, w, l] ≥ s[i, w] - M * (1 - z[i, l])
                m.addConstr(
                    y[i, w, l] >= s[i, w] - M * (1 - z[i, l]),
                    name=f"lin3_i{i}_w{w}_l{l}",
                )

    # Add revenue to objective: -sum_l (price_grid[l] * y[i, w, l])
    for w in W:
        scen = scenarios[w]
        weight = scen.weight
        for i in I:
            obj += weight * (-gp.quicksum(price_grid[l] * y[i, w, l] for l in L))

    m.setObjective(obj, GRB.MINIMIZE)

    var_dict: Dict[str, Any] = {
        "x": x,
        "q": q,
        "s": s,
        "z": z,
        "p": p,
        "y": y,
    }
    return m, var_dict
