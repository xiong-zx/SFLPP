"""
Progressive Hedging algorithm for solving the two-stage stochastic
facility location and pricing problem.
"""

from typing import Dict, Any, Tuple, List
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from .data import Instance
from .extensive_form import ExtensiveForm
from .solver import solve_gurobi_model


def solve_ph_subproblem(
    inst: Instance,
    scen_idx: int,
    ext_form: ExtensiveForm,
    x_bar: Dict[int, float],
    u_multipliers: Dict[int, float],
    rho: float,
    alpha: float,
) -> Tuple[Dict[str, Any], float]:
    """
    Solves the subproblem for a single scenario in the Progressive Hedging algorithm.
    """
    I, J = inst.I, inst.J
    f = inst.f
    a = inst.a
    b = inst.b
    cfg = inst.config
    scen = ext_form.scenarios[scen_idx]

    m = gp.Model(f"PH_subproblem_w{scen_idx}")
    m.setParam("OutputFlag", 0)  # Suppress Gurobi output for subproblems

    # Scenario-specific variables
    x = m.addVars(J, vtype=GRB.BINARY, name="x")
    q = m.addVars(I, J, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    s = m.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="s")
    p = m.addVars(I, lb=cfg.p_min, ub=cfg.p_max, vtype=GRB.CONTINUOUS, name="p")

    # --- Constraints ---
    # s_i = Σ_j q_ij
    m.addConstrs((s[i] == gp.quicksum(q[i, j] for j in J) for i in I), name="s_def")

    # Capacity: Σ_i q_ij ≤ u_j(ω) x_j
    m.addConstrs(
        (gp.quicksum(q[i, j] for i in I) <= scen.u[j] * x[j] for j in J), name="cap"
    )

    # demand fulfillment bounds: (1-alpha) h_i(p_i^w) ≤ s_i^w ≤ h_i(p_i^w)
    for i in I:
        demand_at_price = -a[i] * p[i] + b[i]
        m.addConstr(s[i] <= demand_at_price, name=f"demand_upper_i{i}")
        m.addConstr(s[i] >= (1.0 - alpha) * demand_at_price, name=f"demand_lower_i{i}")

    # --- Objective Function ---
    # Original objective part for this scenario
    obj = gp.LinExpr()
    obj += gp.quicksum(f[j] * x[j] for j in J)  # Fixed cost
    obj += gp.quicksum(
        scen.bar_c[(i, j)] * q[i, j] for i in I for j in J
    )  # Transport cost
    obj -= gp.quicksum(p[i] * s[i] for i in I)  # Revenue (negative)

    # PH terms (linearized quadratic penalty)
    for j in J:
        # Multiplier term: u * x
        obj += u_multipliers[j] * x[j]
        # Quadratic penalty term: (rho/2) * (x - x_bar)^2
        # Linearized as: (rho/2) * (x^2 - 2*x*x_bar + x_bar^2)
        # Since x is binary, x^2 = x. x_bar^2 is a constant and can be ignored.
        obj += (rho / 2.0) * (x[j] * (1.0 - 2.0 * x_bar[j]))

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Subproblem for scenario {scen_idx} failed to solve.")

    # Extract solution
    sol = {
        "x": {j: x[j].X for j in J},
        "p": {i: p[i].X for i in I},
        "s": {i: s[i].X for i in I},
    }
    return sol, m.ObjVal


def solve_with_ph(
    ext_form: ExtensiveForm,
    rho: float,
    max_iter: int = 100,
    alpha: float = 0.1,
    tol: float = 1e-4,
) -> Dict[str, Any]:
    """
    Main Progressive Hedging algorithm loop.
    """
    inst = ext_form.instance
    I, J, W = inst.I, inst.J, ext_form.W
    n_scenarios = len(W)

    # --- Initialization (k=0) ---
    k = 0
    u_multipliers = {w: {j: 0.0 for j in J} for w in W}
    convergence_history = []
    x_scenario = {w: {j: 0.0 for j in J} for w in W}  # Initial feasible x is all zeros
    x_bar = {j: 0.0 for j in J}

    print("--- Starting Progressive Hedging ---")
    print(f"Params: rho={rho}, max_iter={max_iter}, tol={tol}")

    while k < max_iter:
        k += 1
        print(f"\n--- Iteration {k} ---")

        # --- Step 1: Solve subproblems for each scenario ---
        for w in W:
            sol, _ = solve_ph_subproblem(
                inst, w, ext_form, x_bar, u_multipliers[w], rho, alpha
            )
            x_scenario[w] = sol["x"]

        # --- Step 2: Update consensus variable x_bar ---
        x_bar_new = {j: 0.0 for j in J}
        for j in J:
            # Use scenario weights for averaging
            avg_x_j = sum(ext_form.scenarios[w].weight * x_scenario[w][j] for w in W) 
            x_bar_new[j] = avg_x_j
        x_bar = x_bar_new

        # --- Step 3: Update multipliers u ---
        for w in W:
            for j in J:
                u_multipliers[w][j] += rho * (x_scenario[w][j] - x_bar[j])

        # --- Step 4: Check for convergence ---
        max_diff = 0.0
        for w in W:
            for j in J:
                diff = abs(x_scenario[w][j] - x_bar[j])
                if diff > max_diff:
                    max_diff = diff
        convergence_history.append(max_diff)

        avg_x_bar_val = np.mean(list(x_bar.values()))
        print(f"Consensus variable (avg): {avg_x_bar_val:.4f}")
        print(f"Max deviation |x - x_bar|: {max_diff:.6f}")

        if max_diff < tol:
            print(f"\nConvergence reached in {k} iterations.")
            break

    if k == max_iter:
        print("\nWarning: Max iterations reached without convergence.")

    # Post-process: round x_bar to get final integer solution
    final_x = {j: round(val) for j, val in x_bar.items()}

    return {
        "final_x": final_x,
        "x_bar": x_bar,
        "iterations": k,
        "converged": max_diff < tol,
        "convergence_history": convergence_history,
    }


def evaluate_first_stage_solution(
    x_solution: Dict[int, float], ext_form: ExtensiveForm, alpha: float = 0.1
) -> float:
    """
    Given a fixed first-stage solution x, solve the second-stage problems
    for all scenarios to calculate the true objective value.
    """
    inst = ext_form.instance
    I, J, W = inst.I, inst.J, ext_form.W
    f = inst.f
    a = inst.a
    b = inst.b
    cfg = inst.config

    # The model is now just a collection of independent QPs, one for each scenario
    m = gp.Model("evaluation")
    m.setParam("OutputFlag", 0)

    q = m.addVars(I, J, W, lb=0.0, name="q")
    s = m.addVars(I, W, lb=0.0, name="s")
    p = m.addVars(I, W, lb=cfg.p_min, ub=cfg.p_max, name="p")

    # Add all constraints from the original extensive form, but with x fixed
    for w in W:
        scen = ext_form.scenarios[w]
        for i in I:
            m.addConstr(s[i, w] == gp.quicksum(q[i, j, w] for j in J))
        for j in J:
            # Here x_solution is used as a constant
            m.addConstr(gp.quicksum(q[i, j, w] for i in I) <= scen.u[j] * x_solution[j])
        for i in I:
            demand_at_price = -a[i] * p[i, w] + b[i]
            m.addConstr(s[i, w] <= demand_at_price)
            m.addConstr(s[i, w] >= (1.0 - alpha) * demand_at_price)

    # Calculate objective value
    obj = gp.QuadExpr()
    # 1. Fixed cost from the given x_solution
    obj += gp.quicksum(f[j] * x_solution[j] for j in J)

    # 2. Expected recourse cost
    for w in W:
        scen = ext_form.scenarios[w]
        weight = scen.weight
        obj += weight * gp.quicksum(scen.bar_c[(i, j)] * q[i, j, w] for i in I for j in J)
        obj -= weight * gp.quicksum(p[i, w] * s[i, w] for i in I)

    m.setObjective(obj, GRB.MINIMIZE)
    solve_gurobi_model(m)

    if m.Status == GRB.OPTIMAL:
        return m.ObjVal
    raise RuntimeError("Evaluation model failed to solve.")