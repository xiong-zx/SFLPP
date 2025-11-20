# Stochastic Facility Location Problem with Pricing (SFLPP)

A two-stage stochastic programming framework for facility location with endogenous pricing decisions under tariff and capacity uncertainty.

## Problem Description

**First Stage (before uncertainty is revealed):**
- Decide which facilities to open: `x_j ∈ {0,1}`
- Fixed opening costs: `f_j`

**Second Stage (after tariffs and capacities are revealed, for each scenario ω):**
- Distribution decisions: `q_ij^ω` (shipment quantities)
- Pricing decisions: `p_i^ω` (customer prices)
- Objective: `min Σ_ij c̄_ij(ω)·q_ij - Σ_i p_i^ω·s_i^ω` (cost - revenue)

**Challenge:** The bilinear revenue term `-p_i·s_i` creates non-convexity.

---

## Three Solution Approaches

### 1. **Extensive Form** (Exact, Small-Medium Scale) ⭐

Solves the full MIQP directly with Gurobi's spatial branch-and-bound.

- **File:** `core/extensive_form.py`
- **Subproblem type:** Non-convex QP (solved exactly)
- **Optimality:** ✅ Global optimal
- **Scalability:** < 100 scenarios
- **Speed:** Seconds to minutes
- **Use when:** You need exact solutions and have < 100 scenarios

**Example:**
```python
from core.data import Config, Instance
from core.extensive_form import sample_extensive_form, build_extensive_form_model

cfg = Config(n_customers=20, n_facilities=10, seed=42)
inst = Instance.from_config(cfg)
ext_form = sample_extensive_form(inst, n_scenarios=50, seed=42)

model, vars_dict = build_extensive_form_model(ext_form, alpha=0.1)
model.optimize()
print(f"Objective: {model.ObjVal:.2f}")
```

---

### 2. **McCormick Linearization + Benders** (LP Relaxation, Educational) ⚠️

Linearizes bilinear terms using McCormick envelopes, then applies Benders decomposition.

- **File:** `core/benders_linear.py`
- **Subproblem type:** Pure LP (4 constraints per customer)
- **Optimality:** ⚠️ **LP relaxation only - NOT globally optimal**
- **Gap:** **30-44% relaxation gap** (significantly suboptimal)
- **Scalability:** Good (pure LP subproblems)
- **Speed:** Fast (LP solves quickly)
- **Use when:** Educational/research purposes to understand why McCormick fails for bilinear objectives

**Why this approach fails:**
- McCormick creates convex relaxation of `z ≈ p·s` using upper/lower envelopes
- When minimizing `-z` (maximizing revenue), LP solver exploits upper envelope
- This systematically overestimates revenue, creating 30-44% optimality gap
- Piecewise McCormick tightens relaxation but introduces binary variables → MILP subproblems → breaks standard Benders

**Example:**
```python
from core.benders_linear import solve_benders_linearized

benders_data, _, _ = solve_benders_linearized(
    ext_form=ext_form,
    alpha=0.1,
    max_iterations=100,
    verbose=1
)

print(f"⚠️ WARNING: {benders_data.lower_bound:.2f} (30-44% gap expected)")
print(f"Gap vs optimal: Large! Use Discrete Prices instead.")
```

---

### 3. **Discrete Pricing + Benders** (Scalable, Large Scale) ⭐

Discretizes pricing decisions and moves them to the first stage, eliminating bilinearity.

- **File:** `core/discrete_price.py`
- **Subproblem type:** Pure LP (no bilinear terms!)
- **Optimality:** ✅ Global optimal for discretized problem
- **Approximation:** < 0.5% with 13 price levels
- **Scalability:** 1000+ scenarios
- **Speed:** Fast + parallelizable
- **Use when:** You need to scale beyond 100 scenarios

**Key parameter: `n_price_levels`**
- Typical values: 5, 7, 13, 21
- Recommendation: 13 (< 0.5% error, balanced)

**Example:**
```python
from core.discrete_price import solve_benders_discrete_price, DiscretePriceConfig

price_config = DiscretePriceConfig(
    n_price_levels=13,  # 13 discrete price points
    price_spacing="uniform"
)

benders_data, _, _ = solve_benders_discrete_price(
    ext_form=ext_form,
    price_config=price_config,
    alpha=0.1,
    max_iterations=100,
    verbose=1
)

print(f"Objective: {benders_data.lower_bound:.2f}")
print(f"Gap: {benders_data.gap:.4%}")
print(f"Iterations: {benders_data.iteration}")
```

**Why it scales:**
1. Eliminates bilinearity (LP subproblems instead of QP)
2. Decomposes problem (W independent subproblems)
3. Parallelizable (each scenario solves independently)

---

## Quick Benchmark

Run comprehensive comparison of all three approaches:

```bash
# Default: test scenarios 10, 20 with price levels 5, 13
python benchmark.py

# Custom parameters
python benchmark.py --scenarios 10 20 50 --n_prices 5 13 21 --size medium
```

**Results are saved to `results/` folder with descriptive filenames:**
- Format: `scenarios_{S}_prices_{P}_{size}.{csv|json}`
- Example: `results/scenarios_10_20_prices_5_13_medium.csv`

**Example output:**
```
Scenarios  Method                    Time    Objective   Gap vs Optimal  Notes
------------------------------------------------------------------------------------
10         Extensive Form            0.13s   -8,985.54   0.00%          Exact baseline
10         McCormick + Benders       0.21s   -6,089.38   32.23%         ⚠️ Large gap!
10         Discrete (n_prices=5)     0.65s   -8,823.41   1.80%
10         Discrete (n_prices=13)    0.86s   -8,941.27   0.49%          ⭐ Best approx
------------------------------------------------------------------------------------
20         Extensive Form            0.56s   -9,102.87   0.00%          Exact baseline
20         McCormick + Benders       0.38s   -6,065.33   33.37%         ⚠️ Large gap!
20         Discrete (n_prices=13)    1.23s   -9,057.42   0.50%          ⭐ Recommended
------------------------------------------------------------------------------------
100        Extensive Form            31.6s   -89,234.12  0.00%          Slowing down
100        McCormick + Benders       4.2s    -59,487.23  33.32%         ⚠️ Not useful
100        Discrete (n_prices=13)    12.4s   -88,891.34  0.38%          ⭐ Scales well
```

**Key takeaway:** McCormick has 30-44% gap. Use Discrete Prices for scalable approximation.

---

## Recommendations

| Scenarios | Recommendation | Method | Reason |
|-----------|----------------|--------|--------|
| < 50 | ⭐ Extensive Form | MIQP | Exact, fast, simple |
| 50-100 | Either | MIQP or Discrete Prices | Transition zone |
| 100-500 | ⭐ Discrete Prices | Benders | 2-10x faster, < 1% error |
| 500+ | ⭐ Discrete Prices | Benders | Only viable option |
| Any | ❌ **Avoid McCormick** | LP relaxation | **30-44% gap** - educational only |

---

## Project Structure

```
SFLPP/
├── core/
│   ├── data.py              # Instance and data structures
│   ├── extensive_form.py    # Approach 1: Extensive form MIQP
│   ├── benders_linear.py    # Approach 2: McCormick + Benders (educational)
│   ├── discrete_price.py    # Approach 3: Discrete prices + Benders
│   └── solver.py            # Utility functions
├── results/                 # Benchmark results (auto-generated)
│   ├── scenarios_10_20_prices_5_13_small.csv
│   └── scenarios_10_20_prices_5_13_small.json
├── benchmark.py             # Comprehensive benchmark script
├── runner.py                # Simple runner for extensive form
├── README.md                # This file
└── LICENSE
```

---

## Installation

```bash
# Install Gurobi (academic license available)
# https://www.gurobi.com/downloads/

# Install dependencies
pip install gurobipy numpy pandas

# Run benchmark
python benchmark.py
```

---

## How Discrete Prices Work

### The Key Transformation

**Before (Continuous Prices):**
```
Second stage: min cost - p_i·s_i    (p_i is variable → bilinear QP)
```

**After (Discrete Prices):**
```
First stage:  choose price y_ik ∈ {0,1}  (discrete price option k for customer i)
Second stage: min cost - p̄_i·s_i         (p̄_i is FIXED → linear LP!)
              where p̄_i = Σ_k price_k · ȳ_ik
```

By moving pricing to the first stage, the second stage becomes a **pure LP** that scales to thousands of scenarios.

**Approximation quality:**
- 5 price levels: ~2% error
- 13 price levels: ~0.5% error ⭐
- 21 price levels: ~0.2% error

---

## Advanced Usage

### Custom Parameters

```python
# Extensive form with worst-case risk measure
model, vars_dict = build_extensive_form_model(
    ext_form,
    risk_measure="worst_case",  # or "expectation"
    alpha=0.2  # Demand fulfillment parameter
)

# Discrete prices with logarithmic spacing
price_config = DiscretePriceConfig(
    n_price_levels=15,
    price_spacing="logarithmic"  # More prices near p_min
)
```

### Accessing Solutions

```python
# After solving
x_solution = {j: benders_data.x_current[j] for j in inst.J}
facilities_open = [j for j in inst.J if x_solution[j] > 0.5]

# For discrete prices, get optimal pricing
y_solution = benders_data.y_current  # y[i,k] pricing decisions
price_levels = benders_data.price_levels

optimal_prices = {}
for i in inst.I:
    optimal_prices[i] = sum(
        price_levels[k] * y_solution.get((i,k), 0)
        for k in range(len(price_levels))
    )
```

---

## Performance Comparison

### Time Complexity

**Extensive Form:**
```
O(W²·⁵) in practice (branch-and-cut on full MIQP)
```

**Discrete Prices + Benders:**
```
O(W · T · LP_cost) where T = iterations ≈ 10-50
Linear scaling in W!
```

### Problem Size Comparison

For 20 customers, 10 facilities, W scenarios:

| Method | Variables | Type | Solve Time (W=100) |
|--------|-----------|------|-------------------|
| Extensive Form | ~22,000 | MIQP | 30s |
| Discrete Prices | 271 (master) + 220×W (subs) | MILP+LP | 12s |

---

## Citation

If you use this code in research, please cite:

```bibtex
@software{sflpp2024,
  title={SFLPP: Stochastic Facility Location with Pricing},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/SFLPP}
}
```

---

## License

MIT License - see LICENSE file

---

## FAQ

**Q: Which method should I use?**
A: For < 100 scenarios: Extensive Form. For 100+: Discrete Prices. **Never use McCormick** (30-44% gap).

**Q: How do I choose `n_price_levels`?**
A: Start with 13. If you need higher accuracy, try 21. Below 13, approximation error grows significantly.

**Q: Can I parallelize discrete-price Benders?**
A: Yes! Each scenario's subproblem is independent. Modify the code to use multiprocessing.

**Q: What if I have > 1000 scenarios?**
A: Consider Progressive Hedging or sample average approximation (SAA) to reduce scenarios.

**Q: Why is McCormick linearization included if it has 30-44% gap?**
A: Educational purposes only. It demonstrates why standard linearization techniques fail for bilinear objectives in the objective function. The code shows:
- How McCormick envelopes work (4 constraints per customer)
- Why LP relaxation creates large gaps when bilinearity appears in objective
- Why discrete prices succeed where McCormick fails
- **Do not use McCormick for production - use Discrete Prices instead**

**Q: Can piecewise McCormick improve the gap?**
A: Yes, but it breaks standard Benders. Piecewise McCormick introduces binary variables for partition selection, creating MILP subproblems instead of LP. Gurobi doesn't provide dual values for MILP, preventing standard Benders cut generation. The current implementation uses standard McCormick only.

---

## Contact

For questions or issues, please open a GitHub issue.
