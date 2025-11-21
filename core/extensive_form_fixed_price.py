"""
Extensive-form builder with FIRST-STAGE prices (prices fixed across scenarios).

Use this to make results comparable with the Benders solver that chooses a
single price per customer for all scenarios.
"""

from typing import Any, Dict, Tuple
import gurobipy as gp
from gurobipy import GRB

from .extensive_form import ExtensiveForm


def build_extensive_form_fixed_price_model(
    ext_form: ExtensiveForm,
    alpha: float = 0.1,
) -> Tuple[gp.Model, Dict[str, Any]]:
    """
    Build an extensive-form MIQP where prices are first-stage (scenario-invariant).

    Returns:
        model: Gurobi model
        vars:  dict with keys 'x', 'q', 's', 'p'
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must be in [0, 1).")

    inst = ext_form.instance
    I, J, W = ext_form.I, ext_form.J, ext_form.W
    f = inst.f
    a = inst.a
    b = inst.b
    cfg = inst.config
    scenarios = ext_form.scenarios

    m = gp.Model("SCFLP_extensive_fixed_price")
    m.setParam("OutputFlag", 0)

    # first-stage binaries x_j
    x = m.addVars(J, vtype=GRB.BINARY, name="x")

    # first-stage prices p_i (shared across scenarios)
    p = m.addVars(
        I,
        lb=cfg.p_min,
        ub=cfg.p_max,
        vtype=GRB.CONTINUOUS,
        name="p",
    )

    # second-stage continuous q_ij^w, s_i^w
    q = m.addVars(I, J, W, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    s = m.addVars(I, W, lb=0.0, vtype=GRB.CONTINUOUS, name="s")

    # s_i^w = Σ_j q_ij^w
    for w in W:
        for i in I:
            m.addConstr(
                s[i, w] == gp.quicksum(q[i, j, w] for j in J),
                name=f"s_def_i{i}_w{w}",
            )

    # capacity constraints: Σ_i q_ij^w ≤ u_j(w) x_j
    for w in W:
        scen = scenarios[w]
        for j in J:
            m.addConstr(
                gp.quicksum(q[i, j, w] for i in I) <= scen.u[j] * x[j],
                name=f"cap_j{j}_w{w}",
            )

    # demand bounds: (1-alpha) h_i(p_i) ≤ s_i^w ≤ h_i(p_i)
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

    # objective: fixed + expected (transport/tariff - revenue)
    obj = gp.QuadExpr()
    obj += gp.quicksum(f[j] * x[j] for j in J)
    for w in W:
        scen = scenarios[w]
        weight = scen.weight
        for i in I:
            obj += weight * gp.quicksum(scen.bar_c[(i, j)] * q[i, j, w] for j in J)
            obj += weight * (-(p[i] * s[i, w]))

    m.setObjective(obj, GRB.MINIMIZE)

    var_dict: Dict[str, Any] = {
        "x": x,
        "q": q,
        "s": s,
        "p": p,
    }
    return m, var_dict
