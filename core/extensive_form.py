"""
Utilities for extensive-form representation of SFLPP instances. Including:

1. ExtensiveForm data structure
2. Sampling ExtensiveForm from Instance
3. Building Gurobi extensive-form model
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .data import Instance, ScenarioData


@dataclass
class ExtensiveForm:
    """
    A sampled extensive-form approximation:

      - A reference Instance (distribution level)
      - A finite set of scenarios {ω=0,...,W-1}, each with (bar_c, u, weight)
    """

    instance: Instance
    scenarios: List[ScenarioData]

    # ---------- convenience properties ----------
    @property
    def I(self) -> List[int]:
        return self.instance.I

    @property
    def J(self) -> List[int]:
        return self.instance.J

    @property
    def W(self) -> List[int]:
        return list(range(len(self.scenarios)))

    # ---------- IO helpers ----------
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dict.
        """
        scen_list: List[Dict[str, Any]] = []
        for s in self.scenarios:
            scen_dict: Dict[str, Any] = {
                "weight": s.weight,
                "u": {str(j): float(val) for j, val in s.u.items()},
                "bar_c": {f"{i},{j}": float(val) for (i, j), val in s.bar_c.items()},
            }
            scen_list.append(scen_dict)

        return {
            "instance": self.instance.to_dict(),
            "scenarios": scen_list,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExtensiveForm":

        inst = Instance.from_dict(d["instance"])

        scenarios: List[ScenarioData] = []
        for s_dict in d["scenarios"]:
            weight = float(s_dict["weight"])
            u = {int(j): float(val) for j, val in s_dict["u"].items()}
            bar_c_raw = s_dict["bar_c"]
            bar_c: Dict[Tuple[int, int], float] = {}
            for key, val in bar_c_raw.items():
                i_str, j_str = key.split(",")
                bar_c[(int(i_str), int(j_str))] = float(val)
            scenarios.append(ScenarioData(bar_c=bar_c, u=u, weight=weight))

        return cls(instance=inst, scenarios=scenarios)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "ExtensiveForm":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)


def sample_extensive_form(
    inst: Instance,
    n_scenarios: int,
    seed: Optional[int] = None,
) -> ExtensiveForm:
    """
    Given an Instance (distribution level), sample an ExtensiveForm
    with n_scenarios scenarios (SAA / Monte Carlo).
    """
    rng = np.random.default_rng(seed)
    I, J = inst.I, inst.J
    tau_max = inst.tau_max
    cap_fluct = inst.cap_fluctuation
    c = inst.c
    base_capacity = inst.base_capacity

    scenarios: List[ScenarioData] = []
    if n_scenarios <= 0:
        return ExtensiveForm(instance=inst, scenarios=[])

    weight = 1.0 / n_scenarios

    for _ in range(n_scenarios):
        bar_c: Dict[Tuple[int, int], float] = {}
        u: Dict[int, float] = {}

        # tariffs & combined costs
        for i in I:
            for j in J:
                tau = rng.uniform(0.0, tau_max)
                g_ij = float(tau * c[(i, j)])
                bar_c[(i, j)] = c[(i, j)] + g_ij

        # capacities
        for j in J:
            eps = rng.uniform(-cap_fluct, cap_fluct)
            u[j] = base_capacity[j] * (1.0 + eps)

        scenarios.append(ScenarioData(bar_c=bar_c, u=u, weight=weight))

    return ExtensiveForm(instance=inst, scenarios=scenarios)


def build_extensive_form_model(
    ext_form: ExtensiveForm,
    risk_measure: str = "expectation",
    alpha: float = 0.1,
) -> Tuple[gp.Model, Dict[str, Any]]:
    """
    Build an extensive-form MIQP model from an ExtensiveForm object.

    Returns:
        model: Gurobi model
        vars:  dict with keys 'x', 'q', 's', 'p', and optionally 'z_worst'
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

    m = gp.Model("SCFLP_extensive")

    # first-stage binaries x_j
    x = m.addVars(J, vtype=GRB.BINARY, name="x")

    # second-stage continuous q_ij^w, s_i^w, and price decisions p_i^w
    q = m.addVars(I, J, W, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    s = m.addVars(I, W, lb=0.0, vtype=GRB.CONTINUOUS, name="s")
    p = m.addVars(
        I,
        W,
        lb=cfg.p_min,
        ub=cfg.p_max,
        vtype=GRB.CONTINUOUS,
        name="p",
    )

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

    # demand fulfillment bounds: (1-alpha) h_i(p_i^w) ≤ s_i^w ≤ h_i(p_i^w)
    for w in W:
        for i in I:
            demand_at_price = -a[i] * p[i, w] + b[i]
            m.addConstr(
                s[i, w] <= demand_at_price,
                name=f"demand_upper_i{i}_w{w}",
            )
            m.addConstr(
                s[i, w] >= (1.0 - alpha) * demand_at_price,
                name=f"demand_lower_i{i}_w{w}",
            )

    # objective
    if risk_measure == "expectation":
        obj = gp.QuadExpr()
        # fixed cost
        obj += gp.quicksum(f[j] * x[j] for j in J)

        # expected recourse
        for w in W:
            scen = scenarios[w]
            weight = scen.weight
            for i in I:
                # Σ_j bar_c_ij(ω) q_ij^w
                obj += weight * gp.quicksum(scen.bar_c[(i, j)] * q[i, j, w] for j in J)
                # - p_i^w Σ_j q_ij^w
                obj += weight * (-(p[i, w] * s[i, w]))

        m.setObjective(obj, GRB.MINIMIZE)
        z_worst = None

    elif risk_measure == "worst_case":
        z_worst = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z_worst")
        fixed_cost = gp.quicksum(f[j] * x[j] for j in J)

        for w in W:
            scen = scenarios[w]
            scenario_expr = gp.QuadExpr()
            for i in I:
                scenario_expr += gp.quicksum(scen.bar_c[(i, j)] * q[i, j, w] for j in J)
                scenario_expr += -(p[i, w] * s[i, w])

            m.addConstr(z_worst >= scenario_expr, name=f"worstcase_w{w}")

        m.setObjective(fixed_cost + z_worst, GRB.MINIMIZE)

    else:
        raise NotImplementedError(
            f"Unknown risk_measure '{risk_measure}'. "
            "Use 'expectation' or 'worst_case'."
        )

    var_dict: Dict[str, Any] = {
        "x": x,
        "q": q,
        "s": s,
        "p": p,
        "z_worst": z_worst,
    }
    return m, var_dict
