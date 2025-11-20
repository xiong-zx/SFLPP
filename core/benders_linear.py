"""
Benders Decomposition with McCormick Linearization for SFLPP.

This module implements Benders decomposition where the bilinear revenue term
(p_i * s_i) is linearized using standard McCormick envelopes.

IMPORTANT: This creates a relaxation with a 30-44% gap. It is included for
educational/research purposes to demonstrate why linearization of bilinear
OBJECTIVES (as opposed to constraints) is challenging.

For production use, prefer:
- Extensive Form (< 100 scenarios): Exact, globally optimal
- Discrete Prices (100+ scenarios): Eliminates bilinearity, scales well

Key components:
1. SubproblemResultLinear: stores subproblem solution and dual-based gradients
2. BendersDataLinear: manages master problem, cuts, and bounds
3. solve_benders_linearized: main algorithm with standard McCormick
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

from .data import Instance, ScenarioData
from .extensive_form import ExtensiveForm


@dataclass
class SubproblemResultLinear:
    """
    Result from solving a linearized scenario subproblem Q(x̄, ω).

    Since subproblem is LP (not QP), we can extract dual values directly.
    """

    scenario_idx: int
    objective_value: float  # Q(x̄, ω)
    is_feasible: bool

    # Gradient for Benders cut: ∂Q/∂x_j from dual of capacity constraint
    gradient: Dict[int, float]  # j -> ∂Q/∂x_j

    # Dual values (directly from LP)
    duals_capacity: Dict[int, float]  # λ_j for capacity constraints

    # Runtime info
    solve_time: float = 0.0


@dataclass
class BendersCut:
    """
    A single Benders optimality cut.

    Cut form: θ ≥ intercept + Σ_j coefficient[j] * x_j
    """

    iteration: int
    intercept: float
    coefficients: Dict[int, float]
    x_bar: Optional[Dict[int, float]] = None


@dataclass
class BendersDataLinear:
    """
    Manages the state of linearized Benders decomposition algorithm.
    """

    ext_form: ExtensiveForm

    # Algorithm state
    iteration: int = 0
    cuts: List[BendersCut] = field(default_factory=list)

    # Bounds
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    gap: float = np.inf

    # Current solution
    x_current: Optional[Dict[int, float]] = None
    theta_current: Optional[float] = None

    # History
    history_lb: List[float] = field(default_factory=list)
    history_ub: List[float] = field(default_factory=list)
    history_gap: List[float] = field(default_factory=list)
    history_time: List[float] = field(default_factory=list)

    def compute_gap(self) -> float:
        """Compute optimality gap."""
        if self.upper_bound >= 1e10 or self.lower_bound <= -1e10:
            return np.inf
        gap_abs = abs(self.upper_bound - self.lower_bound)
        if gap_abs < 1e-6:
            return 0.0
        denominator = max(abs(self.upper_bound), abs(self.lower_bound), 1.0)
        return gap_abs / denominator

    def update_bounds(self, lb: float, ub: float) -> None:
        """Update bounds and compute gap."""
        self.lower_bound = max(self.lower_bound, lb)
        self.upper_bound = min(self.upper_bound, ub)
        self.gap = self.compute_gap()

    def add_cut(self, cut: BendersCut) -> None:
        """Add a Benders cut."""
        self.cuts.append(cut)

    def log_iteration(self, elapsed_time: float) -> None:
        """Log iteration."""
        self.history_lb.append(self.lower_bound)
        self.history_ub.append(self.upper_bound)
        self.history_gap.append(self.gap)
        self.history_time.append(elapsed_time)


def solve_subproblem_linearized(
    x_bar: Dict[int, float],
    scenario: ScenarioData,
    scenario_idx: int,
    inst: Instance,
    alpha: float = 0.1,
    output_flag: int = 0,
) -> SubproblemResultLinear:
    """
    Solve the LINEARIZED LP subproblem for scenario ω given x̄.

    Uses standard McCormick envelopes (4 constraints per customer) to linearize
    the bilinear revenue term p_i * s_i.

    Original QP objective: Σ_ij c̄_ij·q_ij - Σ_i p_i·s_i
    Linearized LP objective: Σ_ij c̄_ij·q_ij - Σ_i z_i

    Standard McCormick envelopes for z_i ≈ p_i·s_i:
        z_i ≥ p_min·s_i + p_i·s_min - p_min·s_min
        z_i ≥ p_max·s_i + p_i·s_max - p_max·s_max
        z_i ≤ p_min·s_i + p_i·s_max - p_min·s_max
        z_i ≤ p_max·s_i + p_i·s_min - p_max·s_min

    where s_min = 0, s_max = -a_i·p_min + b_i (max demand at min price)

    Returns: SubproblemResultLinear with Q(x̄,ω) and dual-based gradients
    """
    I, J = inst.I, inst.J
    a, b = inst.a, inst.b
    cfg = inst.config
    bar_c = scenario.bar_c
    u = scenario.u

    p_min, p_max = cfg.p_min, cfg.p_max

    # Build subproblem model
    m = gp.Model(f"Subproblem_Linear_{scenario_idx}")
    m.setParam("OutputFlag", output_flag)

    # Variables
    q = m.addVars(I, J, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    s = m.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="s")
    p = m.addVars(I, lb=p_min, ub=p_max, vtype=GRB.CONTINUOUS, name="p")

    # Linearization variables z_i ≈ p_i * s_i
    z = m.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="z")

    # Sales definition: s_i = Σ_j q_ij
    for i in I:
        m.addConstr(
            s[i] == gp.quicksum(q[i, j] for j in J),
            name=f"sales_def_{i}"
        )

    # Capacity constraints: Σ_i q_ij ≤ u_j(ω)·x̄_j
    capacity_constrs = {}
    for j in J:
        capacity_constrs[j] = m.addConstr(
            gp.quicksum(q[i, j] for i in I) <= u[j] * x_bar[j],
            name=f"capacity_{j}"
        )

    # Demand bounds
    for i in I:
        demand_expr = -a[i] * p[i] + b[i]
        m.addConstr(s[i] <= demand_expr, name=f"demand_upper_{i}")
        m.addConstr(s[i] >= (1.0 - alpha) * demand_expr, name=f"demand_lower_{i}")

    # Standard McCormick envelopes for z_i ≈ p_i * s_i
    # (4 linear constraints per customer - pure LP!)
    for i in I:
        s_min_i = 0.0
        s_max_i = -a[i] * p_min + b[i]

        # Standard 4-constraint McCormick envelope
        m.addConstr(z[i] >= p_min * s[i] + p[i] * s_min_i - p_min * s_min_i,
                   name=f"mccormick_{i}_1")
        m.addConstr(z[i] >= p_max * s[i] + p[i] * s_max_i - p_max * s_max_i,
                   name=f"mccormick_{i}_2")
        m.addConstr(z[i] <= p_min * s[i] + p[i] * s_max_i - p_min * s_max_i,
                   name=f"mccormick_{i}_3")
        m.addConstr(z[i] <= p_max * s[i] + p[i] * s_min_i - p_max * s_min_i,
                   name=f"mccormick_{i}_4")

    # LINEARIZED Objective: min Σ_ij c̄_ij·q_ij - Σ_i z_i
    # (Note: z_i approximates p_i·s_i via McCormick)
    obj = gp.LinExpr()
    for i in I:
        for j in J:
            obj += bar_c[(i, j)] * q[i, j]
        obj += -z[i]  # Revenue term (linearized)

    m.setObjective(obj, GRB.MINIMIZE)

    # Solve LP
    start_time = time.time()
    m.optimize()
    solve_time = time.time() - start_time

    # Extract results
    if m.Status == GRB.OPTIMAL:
        obj_value = m.ObjVal

        # Extract dual values from capacity constraints (standard Benders)
        duals_capacity = {}
        gradient = {}

        for j in J:
            # Get dual value λ_j for capacity constraint
            lambda_j = capacity_constrs[j].Pi
            duals_capacity[j] = lambda_j

            # Gradient: ∂Q/∂x_j = -λ_j * u_j(ω)
            # (Negative because increasing x_j increases RHS, which decreases cost)
            gradient[j] = -lambda_j * u[j]

        return SubproblemResultLinear(
            scenario_idx=scenario_idx,
            objective_value=obj_value,
            is_feasible=True,
            gradient=gradient,
            duals_capacity=duals_capacity,
            solve_time=solve_time,
        )

    elif m.Status == GRB.INFEASIBLE:
        # Subproblem infeasible - need feasibility cuts
        print(f"\nWARNING: Linearized subproblem {scenario_idx} is INFEASIBLE")
        print(f"x_bar = {x_bar}")
        return SubproblemResultLinear(
            scenario_idx=scenario_idx,
            objective_value=np.inf,
            is_feasible=False,
            gradient={j: 0.0 for j in J},
            duals_capacity={j: 0.0 for j in J},
            solve_time=solve_time,
        )

    else:
        raise RuntimeError(
            f"Linearized subproblem {scenario_idx} terminated with status {m.Status}"
        )


def build_benders_master(
    inst: Instance,
    initial_theta_lb: float = -1e6,
) -> Tuple[gp.Model, Dict[str, Any]]:
    """
    Build the Benders master problem.

    Master: min Σ_j f_j·x_j + θ
            s.t. θ ≥ initial_theta_lb
                 Σ_j x_j ≥ 1  (at least one facility)
                 x_j ∈ {0,1}

    Cuts added iteratively.
    """
    J = inst.J
    f = inst.f

    m = gp.Model("Benders_Master_Linear")

    # Variables
    x = m.addVars(J, vtype=GRB.BINARY, name="x")
    theta = m.addVar(lb=initial_theta_lb, vtype=GRB.CONTINUOUS, name="theta")

    # Objective
    obj = gp.quicksum(f[j] * x[j] for j in J) + theta
    m.setObjective(obj, GRB.MINIMIZE)

    # At least one facility must be open
    m.addConstr(gp.quicksum(x[j] for j in J) >= 1, name="at_least_one_facility")

    return m, {"x": x, "theta": theta}


def generate_benders_cut(
    subproblem_results: List[SubproblemResultLinear],
    x_bar: Dict[int, float],
    ext_form: ExtensiveForm,
    iteration: int,
    cut_type: str = "single",
) -> List[BendersCut]:
    """
    Generate Benders optimality cuts from linearized subproblem results.

    Single cut: θ ≥ Σ_ω p_ω [Q(x̄,ω) - Σ_j (∂Q/∂x_j)·x̄_j] + Σ_j [Σ_ω p_ω (∂Q/∂x_j)]·x_j

    Multi-cut: One cut per scenario (stronger but more constraints)
    """
    J = ext_form.J
    scenarios = ext_form.scenarios

    if cut_type == "single":
        # Aggregated cut across all scenarios
        intercept = 0.0
        coefficients = {j: 0.0 for j in J}

        for result in subproblem_results:
            weight = scenarios[result.scenario_idx].weight
            Q_val = result.objective_value

            # intercept += weight * [Q(x̄,ω) - Σ_j gradient[j]·x̄_j]
            intercept += weight * Q_val
            for j in J:
                intercept -= weight * result.gradient[j] * x_bar[j]

            # coeff[j] += weight * gradient[j]
            for j in J:
                coefficients[j] += weight * result.gradient[j]

        return [BendersCut(
            iteration=iteration,
            intercept=intercept,
            coefficients=coefficients,
            x_bar=x_bar.copy(),
        )]

    elif cut_type == "multi":
        # One cut per scenario
        cuts = []

        for result in subproblem_results:
            Q_val = result.objective_value

            intercept = Q_val
            coefficients = {}

            for j in J:
                intercept -= result.gradient[j] * x_bar[j]
                coefficients[j] = result.gradient[j]

            cuts.append(BendersCut(
                iteration=iteration,
                intercept=intercept,
                coefficients=coefficients,
                x_bar=x_bar.copy(),
            ))

        return cuts

    else:
        raise ValueError(f"Unknown cut_type: {cut_type}")


def add_cuts_to_master(
    master_model: gp.Model,
    master_vars: Dict[str, Any],
    cuts: List[BendersCut],
) -> None:
    """Add Benders cuts to master problem."""
    x = master_vars["x"]
    theta = master_vars["theta"]
    J = list(x.keys())

    for cut in cuts:
        cut_expr = cut.intercept + gp.quicksum(
            cut.coefficients[j] * x[j] for j in J
        )
        master_model.addConstr(
            theta >= cut_expr,
            name=f"benders_cut_iter_{cut.iteration}"
        )

    master_model.update()


def solve_benders_linearized(
    ext_form: ExtensiveForm,
    alpha: float = 0.1,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    cut_type: str = "single",
    initial_solution: Optional[Dict[int, float]] = None,
    verbose: int = 1,
    master_params: Optional[Dict[str, Any]] = None,
    subproblem_params: Optional[Dict[str, Any]] = None,
) -> Tuple[BendersDataLinear, gp.Model, Dict[str, Any]]:
    """
    Solve SFLPP using LINEARIZED Benders decomposition with LP subproblems.

    This uses standard McCormick envelopes to linearize the bilinear term p*s,
    resulting in pure LP subproblems.

    WARNING: McCormick creates a 30-44% relaxation gap for bilinear objectives.
    This is included for educational/research purposes only.

    Args:
        ext_form: Extensive form problem data
        alpha: Demand fulfillment parameter
        max_iterations: Maximum Benders iterations
        tolerance: Convergence tolerance (relative gap)
        cut_type: "single" or "multi"
        initial_solution: Initial x values (default: all facilities open)
        verbose: 0 (silent), 1 (summary), 2 (detailed)
        master_params: Gurobi parameters for master
        subproblem_params: Gurobi parameters for subproblems

    Returns:
        benders_data: Algorithm state and solution
        master_model: Final master problem
        master_vars: Master variables
    """
    inst = ext_form.instance
    scenarios = ext_form.scenarios
    I, J, W = ext_form.I, ext_form.J, ext_form.W

    # Initialize
    benders_data = BendersDataLinear(ext_form=ext_form)

    # Build master
    master_model, master_vars = build_benders_master(inst, initial_theta_lb=-1e6)

    if master_params:
        for param, value in master_params.items():
            master_model.setParam(param, value)
    if verbose < 2:
        master_model.setParam("OutputFlag", 0)

    subproblem_output = 0 if verbose < 2 else subproblem_params.get("OutputFlag", 0) if subproblem_params else 0

    # Initial solution
    if initial_solution is None:
        initial_solution = {j: 1.0 for j in J}

    if verbose >= 1:
        print("\n" + "="*70)
        print("LINEARIZED BENDERS DECOMPOSITION (McCormick)")
        print("="*70)
        print(f"Instance: {len(I)} customers, {len(J)} facilities, {len(W)} scenarios")
        print(f"Method: Standard McCormick envelope linearization (4 constraints/customer)")
        print(f"Cut type: {cut_type}, Tolerance: {tolerance}")
        print(f"⚠️  WARNING: McCormick creates ~30-44% relaxation gap for bilinear objectives")
        print("="*70)

    start_time = time.time()

    # Generate initial cuts
    if verbose >= 1:
        print("Generating initial cuts...")

    initial_subproblems = []
    for w in W:
        result = solve_subproblem_linearized(
            x_bar=initial_solution,
            scenario=scenarios[w],
            scenario_idx=w,
            inst=inst,
            alpha=alpha,
            output_flag=subproblem_output,
        )
        initial_subproblems.append(result)

    initial_cuts = generate_benders_cut(
        subproblem_results=initial_subproblems,
        x_bar=initial_solution,
        ext_form=ext_form,
        iteration=-1,
        cut_type=cut_type,
    )

    for cut in initial_cuts:
        benders_data.add_cut(cut)

    add_cuts_to_master(master_model, master_vars, initial_cuts)

    # Initial upper bound
    fixed_cost_init = sum(inst.f[j] * initial_solution[j] for j in J)
    expected_recourse_init = sum(
        scenarios[w].weight * initial_subproblems[w].objective_value
        for w in W
    )
    initial_ub = fixed_cost_init + expected_recourse_init
    benders_data.upper_bound = initial_ub

    if verbose >= 1:
        print(f"Initial upper bound: {initial_ub:.2f}")

    # Main Benders loop
    for iteration in range(max_iterations):
        iter_start = time.time()

        # Solve master
        master_model.optimize()

        if master_model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Master not optimal at iteration {iteration}")

        # Extract solution
        x_bar = {j: master_vars["x"][j].X for j in J}
        theta_val = master_vars["theta"].X
        master_obj = master_model.ObjVal

        benders_data.x_current = x_bar
        benders_data.theta_current = theta_val
        benders_data.iteration = iteration

        lower_bound = master_obj

        # Solve subproblems
        subproblem_results = []
        for w in W:
            result = solve_subproblem_linearized(
                x_bar=x_bar,
                scenario=scenarios[w],
                scenario_idx=w,
                inst=inst,
                alpha=alpha,
                output_flag=subproblem_output,
            )
            subproblem_results.append(result)

        # Upper bound
        fixed_cost = sum(inst.f[j] * x_bar[j] for j in J)
        expected_recourse = sum(
            scenarios[w].weight * subproblem_results[w].objective_value
            for w in W
        )
        upper_bound = fixed_cost + expected_recourse

        # Update bounds
        benders_data.update_bounds(lower_bound, upper_bound)

        iter_time = time.time() - iter_start
        benders_data.log_iteration(time.time() - start_time)

        # Log
        if verbose >= 1:
            open_facilities = [j for j in J if x_bar[j] > 0.5]
            print(f"Iter {iteration:3d} | LB: {lower_bound:12.2f} | UB: {upper_bound:12.2f} | "
                  f"Gap: {benders_data.gap:8.2%} | Facilities: {len(open_facilities):2d} | "
                  f"Time: {iter_time:6.2f}s")

        # Check convergence
        if benders_data.gap <= tolerance:
            if verbose >= 1:
                print("="*70)
                print(f"CONVERGED in {iteration + 1} iterations")
                print(f"Final gap: {benders_data.gap:.6f}")
                print(f"Total time: {time.time() - start_time:.2f}s")
                print("="*70)
            break

        # Generate and add cuts
        cuts = generate_benders_cut(
            subproblem_results=subproblem_results,
            x_bar=x_bar,
            ext_form=ext_form,
            iteration=iteration,
            cut_type=cut_type,
        )

        for cut in cuts:
            benders_data.add_cut(cut)

        add_cuts_to_master(master_model, master_vars, cuts)

    else:
        if verbose >= 1:
            print("="*70)
            print(f"MAXIMUM ITERATIONS ({max_iterations}) REACHED")
            print(f"Final gap: {benders_data.gap:.6f}")
            print(f"Total time: {time.time() - start_time:.2f}s")
            print("="*70)

    return benders_data, master_model, master_vars
