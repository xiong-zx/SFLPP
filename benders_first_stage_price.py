"""
Benders Decomposition for Stochastic Facility Location Problem
with First-Stage Pricing Decisions

This implementation solves the two-stage stochastic facility location problem
where both facility locations (x) and prices (p) are first-stage decisions.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ProblemData:
    """Data structure for the facility location problem with pricing"""

    # Sets
    n_customers: int  # number of customers
    n_facilities: int  # number of potential facilities
    n_scenarios: int  # number of scenarios

    # Deterministic parameters
    f: np.ndarray  # f[j]: fixed cost to open facility j
    c: np.ndarray  # c[i,j]: transportation cost from facility j to customer i
    a: np.ndarray  # a[i]: price sensitivity for customer i
    b: np.ndarray  # b[i]: base demand for customer i
    alpha: float   # service level (must serve at least 1-alpha fraction)

    # Stochastic parameters
    g: np.ndarray  # g[omega,i,j]: tariff cost in scenario omega
    u: np.ndarray  # u[omega,j]: capacity of facility j in scenario omega
    pi: np.ndarray  # pi[omega]: probability of scenario omega

    def __post_init__(self):
        """Validate data consistency"""
        assert self.f.shape == (self.n_facilities,)
        assert self.c.shape == (self.n_customers, self.n_facilities)
        assert self.a.shape == (self.n_customers,)
        assert self.b.shape == (self.n_customers,)
        assert self.g.shape == (self.n_scenarios, self.n_customers, self.n_facilities)
        assert self.u.shape == (self.n_scenarios, self.n_facilities)
        assert self.pi.shape == (self.n_scenarios,)
        assert np.isclose(self.pi.sum(), 1.0), "Scenario probabilities must sum to 1"
        assert 0 <= self.alpha <= 1, "Service level must be between 0 and 1"


class BendersDecomposition:
    """
    Benders Decomposition solver for the facility location problem
    with first-stage pricing decisions
    """

    def __init__(self, data: ProblemData, verbose: bool = True,
                 price_levels: int = 3, price_grid: Optional[List[np.ndarray]] = None):
        self.data = data
        self.verbose = verbose
        self.price_levels = price_levels

        # Build a discrete price grid for each customer if none is provided
        if price_grid is None:
            self.price_grid = self._build_price_grid(price_levels)
        else:
            self.price_grid = price_grid

        # Precompute revenue for each (customer, price level): p * (b - a p)
        self.revenue_matrix = []
        for i in range(self.data.n_customers):
            rev_i = []
            for p_val in self.price_grid[i]:
                demand_i = max(0.0, self.data.b[i] - self.data.a[i] * p_val)
                rev_i.append(p_val * demand_i)
            self.revenue_matrix.append(np.array(rev_i))

        # Storage for Benders cuts
        self.cuts = {omega: [] for omega in range(data.n_scenarios)}

        # Iteration tracking
        self.iteration = 0
        self.lower_bounds = []
        self.upper_bounds = []

        # Best solution found
        self.best_x = None
        self.best_p = None
        self.best_obj = float('inf')

    def _build_price_grid(self, levels: int) -> List[np.ndarray]:
        """
        Create an evenly spaced price grid between p_min and p_max for each customer.
        """
        grid = []
        for i in range(self.data.n_customers):
            p_max = self.data.b[i] / self.data.a[i]
            p_min = max(1.0, np.mean(self.data.c[i, :]))
            grid.append(np.linspace(p_min, p_max, levels))
        return grid

    def _initialize_cuts(self):
        """
        Initialize with Benders cuts from a simple initial solution
        """
        # Use a simple initial solution: open all facilities, pick middle price level
        x_init = np.ones(self.data.n_facilities)
        p_init = np.array([self.price_grid[i][len(self.price_grid[i]) // 2]
                           for i in range(self.data.n_customers)])

        if self.verbose:
            print("Initializing with cuts from a simple solution...")

        # Generate cuts for all scenarios
        for omega in range(self.data.n_scenarios):
            result = self._solve_subproblem_dual(x_init, p_init, omega)
            if result['feasible']:
                self._add_benders_cut(omega, result['dual_sol'])
            else:
                # If even the initial solution is infeasible, we have a problem
                raise RuntimeError(f"Initial solution is infeasible for scenario {omega}. "
                                 "Problem may be fundamentally infeasible.")

        if self.verbose:
            print(f"Added {self.data.n_scenarios} initial cuts (one per scenario)\n")

    def solve(self, max_iter: int = 100, tol: float = 1e-4) -> Dict:
        """
        Main Benders decomposition algorithm

        Parameters:
        -----------
        max_iter: Maximum number of iterations
        tol: Convergence tolerance for optimality gap

        Returns:
        --------
        Dictionary containing solution and convergence information
        """

        if self.verbose:
            print("=" * 80)
            print("BENDERS DECOMPOSITION - FIRST-STAGE PRICING")
            print("=" * 80)
            print(f"Problem size: {self.data.n_customers} customers, "
                  f"{self.data.n_facilities} facilities, {self.data.n_scenarios} scenarios")
            print("-" * 80)

        # Initialize with cuts from a simple solution
        self._initialize_cuts()

        for iteration in range(max_iter):
            self.iteration = iteration

            # Step 1: Solve master problem
            x_curr, p_curr, theta_curr, master_obj = self._solve_master()

            if x_curr is None:
                print("Master problem infeasible!")
                return None

            # Step 2: Solve subproblems and generate cuts
            total_recourse = 0.0
            cuts_added = 0
            # Expected revenue under current prices (deterministic across scenarios)
            revenue_curr = np.sum([
                p_curr[i] * max(0.0, self.data.b[i] - self.data.a[i] * p_curr[i])
                for i in range(self.data.n_customers)
            ])

            for omega in range(self.data.n_scenarios):
                # Solve subproblem and get result
                result = self._solve_subproblem_dual(x_curr, p_curr, omega)

                if result['feasible']:
                    # Subproblem is feasible - add optimality cut if needed
                    Q_omega = result['obj_value']
                    dual_sol = result['dual_sol']

                    # Check if cut is needed (theta[omega] < Q(x,p;omega) - tol)
                    if theta_curr[omega] < Q_omega - tol:
                        # Generate and add Benders optimality cut
                        self._add_benders_cut(omega, dual_sol)
                        cuts_added += 1

                    total_recourse += self.data.pi[omega] * Q_omega
                else:
                    # Subproblem is infeasible - add feasibility cut
                    extreme_ray = result['extreme_ray']
                    self._add_feasibility_cut(omega, extreme_ray)
                    cuts_added += 1
                    if self.verbose:
                        print(f"  Scenario {omega}: INFEASIBLE - added feasibility cut")
                    # Treat as very large cost for upper bound calculation
                    total_recourse += self.data.pi[omega] * 1e10

            # Step 3: Update bounds
            lower_bound = master_obj
            fixed_cost = np.sum(self.data.f * x_curr)
            upper_bound = fixed_cost + total_recourse - revenue_curr

            self.lower_bounds.append(lower_bound)
            self.upper_bounds.append(upper_bound)

            # Update best solution
            if upper_bound < self.best_obj:
                self.best_obj = upper_bound
                self.best_x = x_curr.copy()
                self.best_p = p_curr.copy()

            # Calculate optimality gap
            denom = max(abs(upper_bound), 1e-8)
            gap = abs(upper_bound - lower_bound) / denom

            if self.verbose:
                print(f"Iter {iteration:3d}: LB={lower_bound:12.4f}, UB={upper_bound:12.4f}, "
                      f"Gap={gap:8.2%}, Cuts={cuts_added}")

            # Step 4: Check convergence
            if gap < tol and cuts_added == 0:
                if self.verbose:
                    print("-" * 80)
                    print(f"Converged in {iteration + 1} iterations!")
                    print(f"Optimal objective: {self.best_obj:.4f}")
                break
        else:
            if self.verbose:
                print("-" * 80)
                print(f"Reached maximum iterations ({max_iter})")
                print(f"Best objective: {self.best_obj:.4f}, Gap: {gap:.2%}")

        return {
            'objective': self.best_obj,
            'x': self.best_x,
            'p': self.best_p,
            'iterations': self.iteration + 1,
            'lower_bounds': self.lower_bounds,
            'upper_bounds': self.upper_bounds,
            'gap': gap
        }

    def _solve_master(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Solve the Benders master problem

        Returns:
        --------
        x: Facility opening decisions
        p: Pricing decisions
        theta: Recourse cost estimates
        obj: Objective value
        """

        model = gp.Model("BendersMaster")
        if not self.verbose:
            model.setParam('OutputFlag', 0)

        n_customers = self.data.n_customers
        n_facilities = self.data.n_facilities
        n_scenarios = self.data.n_scenarios

        # Decision variables
        x = model.addVars(n_facilities, vtype=GRB.BINARY, name="x")
        # Price selection binaries: y[i,k] = 1 if customer i uses price level k
        y = {(i, k): model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}")
             for i in range(n_customers) for k in range(len(self.price_grid[i]))}
        p = model.addVars(n_customers, lb=0.0, name="p")  # implied by y, but kept for cuts
        theta = model.addVars(n_scenarios, lb=-GRB.INFINITY, name="theta")

        # Objective: minimize fixed costs + expected recourse - revenue
        obj_expr = gp.quicksum(self.data.f[j] * x[j] for j in range(n_facilities))
        obj_expr += gp.quicksum(self.data.pi[omega] * theta[omega]
                                for omega in range(n_scenarios))
        obj_expr -= gp.quicksum(self.revenue_matrix[i][k] * y[i, k]
                                for i in range(n_customers)
                                for k in range(len(self.price_grid[i])))
        model.setObjective(obj_expr, GRB.MINIMIZE)

        # Exactly one price level per customer and link p to chosen level
        for i in range(n_customers):
            model.addConstr(gp.quicksum(y[i, k] for k in range(len(self.price_grid[i]))) == 1,
                            name=f"one_price_{i}")
            price_expr = gp.quicksum(self.price_grid[i][k] * y[i, k]
                                     for k in range(len(self.price_grid[i])))
            model.addConstr(p[i] == price_expr, name=f"link_p_{i}")

        # Force at least one facility to open (to ensure feasibility)
        model.addConstr(gp.quicksum(x[j] for j in range(n_facilities)) >= 1,
                       name="min_one_facility")

        # Ensure sufficient capacity: total capacity >= (1-alpha) * total demand
        # Use minimum capacity across scenarios and minimum demand (at max price)
        min_demand_per_customer = np.array([max(0, self.data.b[i] - self.data.a[i] * (self.data.b[i] / self.data.a[i]))
                                            for i in range(n_customers)])
        total_min_demand = np.sum(min_demand_per_customer)

        # Use average capacity across scenarios for planning
        avg_capacity = np.mean(self.data.u, axis=0)

        # Require: sum_j (avg_capacity_j * x_j) >= (1-alpha) * estimated_demand
        # Estimate demand at mid-range prices
        estimated_demand = np.sum([self.data.b[i] - self.data.a[i] * (self.data.b[i] / self.data.a[i] / 2)
                                   for i in range(n_customers)])

        model.addConstr(gp.quicksum(avg_capacity[j] * x[j] for j in range(n_facilities)) >=
                       (1 - self.data.alpha) * estimated_demand,
                       name="min_capacity")

        # Add Benders cuts
        for omega in range(n_scenarios):
            for cut_idx, cut_coef in enumerate(self.cuts[omega]):
                # Compute cut expression
                cut_expr = cut_coef['intercept']
                cut_expr += gp.quicksum(cut_coef['p_coef'][i] * p[i]
                                       for i in range(n_customers))
                cut_expr += gp.quicksum(cut_coef['x_coef'][j] * x[j]
                                       for j in range(n_facilities))

                # Check if this is a feasibility cut or optimality cut
                if cut_coef.get('feasibility_cut', False):
                    # Feasibility cut: 0 >= cut_expr, or equivalently cut_expr <= 0
                    model.addConstr(cut_expr <= 0,
                                   name=f"feas_cut_{omega}_{cut_idx}")
                else:
                    # Optimality cut: theta[omega] >= cut_expr
                    model.addConstr(theta[omega] >= cut_expr,
                                   name=f"opt_cut_{omega}_{cut_idx}")

        # Solve
        model.optimize()

        if model.status != GRB.OPTIMAL:
            return None, None, None, None

        # Extract solution
        x_sol = np.array([x[j].X for j in range(n_facilities)])
        p_sol = np.array([p[i].X for i in range(n_customers)])
        theta_sol = np.array([theta[omega].X for omega in range(n_scenarios)])

        return x_sol, p_sol, theta_sol, model.ObjVal

    def _solve_subproblem_dual(self, x: np.ndarray, p: np.ndarray, omega: int) -> Dict:
        """
        Solve the second-stage subproblem (primal form) and return solution info

        Returns a dictionary with:
        - 'feasible': True if subproblem is feasible
        - 'obj_value': Optimal objective value (if feasible)
        - 'dual_sol': Dual solution (if feasible)
        - 'extreme_ray': Extreme ray from infeasibility (if infeasible)
        """

        model = gp.Model("Subproblem_Primal")
        model.setParam('OutputFlag', 0)
        model.setParam('InfUnbdInfo', 1)  # Request infeasibility information

        n_customers = self.data.n_customers
        n_facilities = self.data.n_facilities

        # Variables: q[i,j] = flow from facility j to customer i
        q = model.addVars(n_customers, n_facilities, lb=0.0, name="q")

        # Objective: minimize cost
        obj_expr = 0.0
        for i in range(n_customers):
            for j in range(n_facilities):
                cost_ij = self.data.c[i, j] + self.data.g[omega, i, j]
                obj_expr += cost_ij * q[i, j]

        model.setObjective(obj_expr, GRB.MINIMIZE)

        # Demand constraints
        demand_lb_constrs = []
        demand_ub_constrs = []
        for i in range(n_customers):
            demand_i = self.data.b[i] - self.data.a[i] * p[i]
            total_flow = gp.quicksum(q[i, j] for j in range(n_facilities))

            # Lower bound: serve at least (1-alpha) fraction
            c_lb = model.addConstr(total_flow >= (1 - self.data.alpha) * demand_i,
                                  name=f"demand_lb_{i}")
            demand_lb_constrs.append(c_lb)

            # Upper bound: serve at most full demand
            c_ub = model.addConstr(total_flow <= demand_i,
                                  name=f"demand_ub_{i}")
            demand_ub_constrs.append(c_ub)

        # Capacity constraints
        capacity_constrs = []
        for j in range(n_facilities):
            total_from_j = gp.quicksum(q[i, j] for i in range(n_customers))
            c_cap = model.addConstr(total_from_j <= self.data.u[omega, j] * x[j],
                                   name=f"capacity_{j}")
            capacity_constrs.append(c_cap)

        model.optimize()

        if model.status == GRB.OPTIMAL:
            # Extract dual variables (constraint shadow prices)
            dual_sol = {
                'lambda': np.array([demand_lb_constrs[i].Pi for i in range(n_customers)]),
                'mu': np.array([demand_ub_constrs[i].Pi for i in range(n_customers)]),
                'nu': np.array([capacity_constrs[j].Pi for j in range(n_facilities)])
            }

            return {
                'feasible': True,
                'obj_value': model.ObjVal,
                'dual_sol': dual_sol
            }

        elif model.status == GRB.INFEASIBLE:
            # Get Farkas dual (extreme ray proving infeasibility)
            extreme_ray = {
                'lambda': np.array([demand_lb_constrs[i].FarkasDual for i in range(n_customers)]),
                'mu': np.array([demand_ub_constrs[i].FarkasDual for i in range(n_customers)]),
                'nu': np.array([capacity_constrs[j].FarkasDual for j in range(n_facilities)])
            }

            return {
                'feasible': False,
                'extreme_ray': extreme_ray
            }

        else:
            # Handle other statuses
            raise RuntimeError(f"Unexpected subproblem status {model.status} for scenario {omega}")

    def _add_benders_cut(self, omega: int, dual_sol: Dict):
        """
        Add a Benders optimality cut for scenario omega

        Cut form:
        theta[omega] >= B^omega_k(x, p)

        where B^omega_k(x,p) = intercept + sum_i p_coef[i]*p[i] + sum_j x_coef[j]*x[j]

        B^omega_k(x,p) =
            sum_i (1-alpha)*b_i * lambda_i^{omega,k}
          - sum_i b_i * mu_i^{omega,k}
          + sum_i (-(1-alpha)*a_i * lambda_i^{omega,k} + a_i * mu_i^{omega,k}) * p_i
          - sum_j u_j^omega * nu_j^{omega,k} * x_j
        """

        n_customers = self.data.n_customers
        n_facilities = self.data.n_facilities

        lam = dual_sol['lambda']
        mu = dual_sol['mu']
        nu = dual_sol['nu']

        # Compute coefficients
        # Intercept: constant terms
        intercept = 0.0
        for i in range(n_customers):
            intercept += (1 - self.data.alpha) * self.data.b[i] * lam[i]
            intercept += self.data.b[i] * mu[i]

        # Coefficients for p[i]
        p_coef = np.zeros(n_customers)
        for i in range(n_customers):
            p_coef[i] = -(1 - self.data.alpha) * self.data.a[i] * lam[i]
            p_coef[i] -= self.data.a[i] * mu[i]

        # Coefficients for x[j]
        x_coef = np.zeros(n_facilities)
        for j in range(n_facilities):
            x_coef[j] = self.data.u[omega, j] * nu[j]

        # Store cut
        self.cuts[omega].append({
            'intercept': intercept,
            'p_coef': p_coef,
            'x_coef': x_coef
        })

    def _add_feasibility_cut(self, omega: int, extreme_ray: Dict):
        """
        Add a Benders feasibility cut for scenario omega

        Feasibility cut form (using extreme ray):
        0 >= intercept + sum_i p_coef[i]*p[i] + sum_j x_coef[j]*x[j]

        This is added as a constraint (not involving theta) to rule out
        infeasible combinations of (x, p)
        """

        n_customers = self.data.n_customers
        n_facilities = self.data.n_facilities

        lam = extreme_ray['lambda']
        mu = extreme_ray['mu']
        nu = extreme_ray['nu']

        if self.verbose and self.iteration < 2:  # Only print for first few iterations
            print(f"    Extreme ray: lambda={lam}, mu={mu}, nu={nu}")

        # Compute coefficients (same form as optimality cut)
        intercept = 0.0
        for i in range(n_customers):
            intercept += (1 - self.data.alpha) * self.data.b[i] * lam[i]
            intercept += self.data.b[i] * mu[i]

        p_coef = np.zeros(n_customers)
        for i in range(n_customers):
            p_coef[i] = -(1 - self.data.alpha) * self.data.a[i] * lam[i]
            p_coef[i] -= self.data.a[i] * mu[i]

        x_coef = np.zeros(n_facilities)
        for j in range(n_facilities):
            x_coef[j] = self.data.u[omega, j] * nu[j]

        if self.verbose and self.iteration < 2:
            print(f"    Cut: {intercept:.4f} + sum(p_coef*p) + sum(x_coef*x) <= 0")
            print(f"    p_coef={p_coef}")
            print(f"    x_coef={x_coef}")

        # Store feasibility cut (marked with a flag)
        self.cuts[omega].append({
            'intercept': intercept,
            'p_coef': p_coef,
            'x_coef': x_coef,
            'feasibility_cut': True
        })

    def evaluate_solution(self, x: np.ndarray, p: np.ndarray) -> Dict:
        """
        Evaluate the objective value and subproblem solutions for a given (x, p)

        Returns detailed information about the solution quality
        """

        # Fixed costs
        fixed_cost = np.sum(self.data.f * x)

        # Expected recourse cost
        recourse_costs = np.zeros(self.data.n_scenarios)
        revenue_per_scenario = np.zeros(self.data.n_scenarios)
        scenario_details = []

        for omega in range(self.data.n_scenarios):
            result = self._solve_subproblem_dual(x, p, omega)

            if not result['feasible']:
                return {
                    'error': 'Subproblem infeasible',
                    'omega': omega
                }

            Q_omega = result['obj_value']
            recourse_costs[omega] = Q_omega

            # Also get actual flows from result
            q_sol = self._solve_subproblem_primal(x, p, omega)
            if q_sol is not None:
                served = q_sol.sum(axis=1)
                revenue = np.dot(p, served)
            else:
                served = None
                revenue = 0.0

            revenue_per_scenario[omega] = revenue

            scenario_details.append({
                'omega': omega,
                'probability': self.data.pi[omega],
                'recourse_cost': Q_omega,
                'revenue': revenue,
                'flows': q_sol
            })

        expected_recourse = np.sum(self.data.pi * recourse_costs)
        expected_revenue = np.sum(self.data.pi * revenue_per_scenario)
        total_cost = fixed_cost + expected_recourse - expected_revenue

        return {
            'total_cost': total_cost,
            'fixed_cost': fixed_cost,
            'expected_recourse': expected_recourse,
            'expected_revenue': expected_revenue,
            'x': x,
            'p': p,
            'scenario_details': scenario_details
        }

    def _solve_subproblem_primal(self, x: np.ndarray, p: np.ndarray, omega: int) -> np.ndarray:
        """
        Solve the primal second-stage subproblem to get actual flows q
        """

        model = gp.Model("Subproblem_Primal")
        model.setParam('OutputFlag', 0)

        n_customers = self.data.n_customers
        n_facilities = self.data.n_facilities

        # Variables: q[i,j] = flow from facility j to customer i
        q = model.addVars(n_customers, n_facilities, lb=0.0, name="q")

        # Objective: minimize cost
        obj_expr = 0.0
        for i in range(n_customers):
            for j in range(n_facilities):
                cost_ij = self.data.c[i, j] + self.data.g[omega, i, j]
                obj_expr += cost_ij * q[i, j]

        model.setObjective(obj_expr, GRB.MINIMIZE)

        # Demand constraints
        for i in range(n_customers):
            demand_i = self.data.b[i] - self.data.a[i] * p[i]
            total_flow = gp.quicksum(q[i, j] for j in range(n_facilities))

            # Lower bound: serve at least (1-alpha) fraction
            model.addConstr(total_flow >= (1 - self.data.alpha) * demand_i,
                          name=f"demand_lb_{i}")

            # Upper bound: serve at most full demand
            model.addConstr(total_flow <= demand_i,
                          name=f"demand_ub_{i}")

        # Capacity constraints
        for j in range(n_facilities):
            total_from_j = gp.quicksum(q[i, j] for i in range(n_customers))
            model.addConstr(total_from_j <= self.data.u[omega, j] * x[j],
                          name=f"capacity_{j}")

        model.optimize()

        if model.status != GRB.OPTIMAL:
            return None

        # Extract solution
        q_sol = np.array([[q[i, j].X for j in range(n_facilities)]
                         for i in range(n_customers)])

        return q_sol


def generate_test_instance(n_customers: int = 5,
                          n_facilities: int = 3,
                          n_scenarios: int = 3,
                          seed: int = 42) -> ProblemData:
    """
    Generate a test instance of the problem
    """

    np.random.seed(seed)

    # Fixed costs for facilities
    f = np.random.uniform(100, 500, n_facilities)

    # Transportation costs (distance-based)
    c = np.random.uniform(1, 10, (n_customers, n_facilities))

    # Demand parameters
    a = np.random.uniform(0.5, 2.0, n_customers)  # price sensitivity
    b = np.random.uniform(50, 150, n_customers)   # base demand

    # Service level
    alpha = 0.1  # must serve at least 90% of demand

    # Scenario probabilities
    pi = np.ones(n_scenarios) / n_scenarios

    # Stochastic tariffs
    g = np.random.uniform(0, 5, (n_scenarios, n_customers, n_facilities))

    # Stochastic capacities
    # Make each facility have enough capacity to serve all demand (to ensure feasibility)
    max_demand = np.sum(b)  # Maximum possible demand (at p=0)
    # Each facility can serve the full demand (very generous for demonstration)
    u = np.random.uniform(0.8, 1.2, (n_scenarios, n_facilities)) * max_demand

    return ProblemData(
        n_customers=n_customers,
        n_facilities=n_facilities,
        n_scenarios=n_scenarios,
        f=f,
        c=c,
        a=a,
        b=b,
        alpha=alpha,
        g=g,
        u=u,
        pi=pi
    )


if __name__ == "__main__":
    # Generate test instance
    print("Generating test instance...")
    data = generate_test_instance(n_customers=5, n_facilities=3, n_scenarios=3)

    print(f"\nProblem parameters:")
    print(f"  Customers: {data.n_customers}")
    print(f"  Facilities: {data.n_facilities}")
    print(f"  Scenarios: {data.n_scenarios}")
    print(f"  Service level: {1-data.alpha:.1%}")
    print(f"\n  Fixed costs: {data.f}")
    print(f"  Base demands: {data.b}")
    print(f"  Price sensitivities: {data.a}")

    # Solve using Benders decomposition
    solver = BendersDecomposition(data, verbose=True)
    result = solver.solve(max_iter=20, tol=1e-4)

    if result is not None:
        print("\n" + "=" * 80)
        print("SOLUTION")
        print("=" * 80)
        print(f"Optimal objective: {result['objective']:.4f}")
        print(f"Iterations: {result['iterations']}")
        print(f"Optimality gap: {result['gap']:.2%}")
        print(f"\nFacility decisions:")
        for j in range(data.n_facilities):
            if result['x'][j] > 0.5:
                print(f"  Facility {j}: OPEN (cost = {data.f[j]:.2f})")
            else:
                print(f"  Facility {j}: CLOSED")

        print(f"\nPricing decisions:")
        for i in range(data.n_customers):
            p_max = data.b[i] / data.a[i]
            demand = max(0, data.b[i] - data.a[i] * result['p'][i])
            print(f"  Customer {i}: p = {result['p'][i]:.2f} (max = {p_max:.2f}), "
                  f"demand = {demand:.2f}")

        # Detailed evaluation
        print(f"\n" + "=" * 80)
        print("DETAILED EVALUATION")
        print("=" * 80)
        eval_result = solver.evaluate_solution(result['x'], result['p'])
        if 'error' in eval_result:
            print(f"ERROR: {eval_result['error']} in scenario {eval_result.get('omega', 'unknown')}")
            print("The solution has infeasible subproblems!")
        else:
            print(f"Total cost: {eval_result['total_cost']:.4f}")
            print(f"  Fixed cost: {eval_result['fixed_cost']:.4f}")
            print(f"  Expected recourse: {eval_result['expected_recourse']:.4f}")
