"""
Comprehensive Benchmark: Comparing Solution Methods for SFLPP

This script compares three solution methods:
1. Extensive Form (exact MIQP)
2. McCormick Linearization + Benders (LP relaxation, ~30-44% gap)
3. Discrete Pricing + Benders (with adjustable price levels)

Usage:
    python benchmark.py --scenarios 10 20 --n_prices 5 13
"""

import argparse
import time
import os
import pandas as pd
from typing import List, Dict
import json

from core.data import Config, Instance
from core.extensive_form import sample_extensive_form, build_extensive_form_model
from core.discrete_price import solve_benders_discrete_price, DiscretePriceConfig
from core.benders_linear import solve_benders_linearized
from core.progressive_hedging import solve_with_ph, evaluate_first_stage_solution


def run_extensive_form(
    inst: Instance,
    n_scenarios: int,
    seed: int = 42,
    verbose: int = 0,
) -> Dict:
    """Run extensive form approach."""
    print(f"\n{'='*70}")
    print(f"EXTENSIVE FORM: {n_scenarios} scenarios")
    print(f"{'='*70}")

    # Sample scenarios
    ext_form = sample_extensive_form(inst, n_scenarios=n_scenarios, seed=seed)

    # Build and solve
    start_time = time.time()
    model, vars_dict = build_extensive_form_model(ext_form, alpha=0.1)
    model.setParam("OutputFlag", verbose)
    model.optimize()
    solve_time = time.time() - start_time

    if model.Status == 2:  # Optimal
        objective = model.ObjVal
        x_sol = {j: vars_dict['x'][j].X for j in inst.J}
        n_facilities = sum(1 for v in x_sol.values() if v > 0.5)

        print(f"Status: OPTIMAL")
        print(f"Objective: {objective:.2f}")
        print(f"Facilities: {n_facilities}")
        print(f"Time: {solve_time:.2f}s")

        return {
            'method': 'Extensive Form',
            'n_scenarios': n_scenarios,
            'status': 'optimal',
            'objective': objective,
            'time': solve_time,
            'n_facilities': n_facilities,
        }
    else:
        print(f"Status: FAILED (status {model.Status})")
        return {
            'method': 'Extensive Form',
            'n_scenarios': n_scenarios,
            'status': 'failed',
            'objective': None,
            'time': solve_time,
        }


def run_discrete_price_benders(
    inst: Instance,
    n_scenarios: int,
    n_price_levels: int = 13,
    seed: int = 42,
    max_iterations: int = 100,
    verbose: int = 0,
) -> Dict:
    """Run discrete pricing + Benders approach."""
    print(f"\n{'='*70}")
    print(f"DISCRETE PRICES + BENDERS: {n_scenarios} scenarios, {n_price_levels} price levels")
    print(f"{'='*70}")

    # Sample scenarios
    ext_form = sample_extensive_form(inst, n_scenarios=n_scenarios, seed=seed)

    # Configure discrete prices
    price_config = DiscretePriceConfig(
        n_price_levels=n_price_levels,
        price_spacing="uniform"
    )

    start_time = time.time()
    try:
        benders_data, master_model, master_vars = solve_benders_discrete_price(
            ext_form=ext_form,
            price_config=price_config,
            alpha=0.1,
            max_iterations=max_iterations,
            tolerance=1e-4,
            verbose=verbose,
        )
        solve_time = time.time() - start_time

        objective = benders_data.lower_bound
        x_sol = benders_data.x_current
        n_facilities = sum(1 for v in x_sol.values() if v > 0.5) if x_sol else 0
        gap = benders_data.gap

        print(f"Status: CONVERGED")
        print(f"Objective: {objective:.2f}")
        print(f"Gap: {gap:.4%}")
        print(f"Facilities: {n_facilities}")
        print(f"Iterations: {benders_data.iteration}")
        print(f"Time: {solve_time:.2f}s")

        return {
            'method': 'Discrete Prices + Benders',
            'n_scenarios': n_scenarios,
            'n_price_levels': n_price_levels,
            'status': 'converged',
            'objective': objective,
            'gap': gap,
            'time': solve_time,
            'iterations': benders_data.iteration,
            'n_facilities': n_facilities,
        }
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"Status: FAILED ({str(e)})")
        return {
            'method': 'Discrete Prices + Benders',
            'n_scenarios': n_scenarios,
            'n_price_levels': n_price_levels,
            'status': 'failed',
            'objective': None,
            'time': solve_time,
            'error': str(e),
        }


def run_progressive_hedging(
    inst: Instance,
    n_scenarios: int,
    rho: float = 1000.0,
    seed: int = 42,
    max_iterations: int = 100,
    verbose: int = 0,
) -> Dict:
    """Run Progressive Hedging approach."""
    print(f"\n{'='*70}")
    print(f"PROGRESSIVE HEDGING: {n_scenarios} scenarios, rho={rho}")
    print(f"{'='*70}")

    # Sample scenarios
    ext_form = sample_extensive_form(inst, n_scenarios=n_scenarios, seed=seed)

    start_time = time.time()
    try:
        ph_results = solve_with_ph(
            ext_form=ext_form,
            rho=rho,
            max_iter=max_iterations,
            alpha=0.1,  # Using the same alpha as other methods
            tol=1e-4,
        )
        solve_time = time.time() - start_time

        # Evaluate the final integer solution to get the true objective value
        final_x = ph_results["final_x"]
        objective = evaluate_first_stage_solution(final_x, ext_form, alpha=0.1)
        n_facilities = sum(1 for v in final_x.values() if v > 0.5)

        print(f"Status: {'CONVERGED' if ph_results['converged'] else 'MAX_ITER'}")
        print(f"Objective: {objective:.2f}")
        print(f"Facilities: {n_facilities}")
        print(f"Iterations: {ph_results['iterations']}")
        print(f"Time: {solve_time:.2f}s")

        return {
            'method': 'Progressive Hedging',
            'n_scenarios': n_scenarios,
            'rho': rho,
            'status': 'converged' if ph_results['converged'] else 'max_iter',
            'objective': objective,
            'time': solve_time,
            'iterations': ph_results['iterations'],
            'n_facilities': n_facilities,
        }
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"Status: FAILED ({str(e)})")
        return {
            'method': 'Progressive Hedging',
            'n_scenarios': n_scenarios,
            'rho': rho,
            'status': 'failed',
            'objective': None,
            'time': solve_time,
            'error': str(e),
        }


def run_mccormick_benders(
    inst: Instance,
    n_scenarios: int,
    seed: int = 42,
    max_iterations: int = 100,
    verbose: int = 0,
) -> Dict:
    """Run McCormick linearization + Benders approach."""
    print(f"\n{'='*70}")
    print(f"MCCORMICK + BENDERS: {n_scenarios} scenarios")
    print(f"⚠️  WARNING: McCormick creates ~30-44% relaxation gap")
    print(f"{'='*70}")

    # Sample scenarios
    ext_form = sample_extensive_form(inst, n_scenarios=n_scenarios, seed=seed)

    start_time = time.time()
    try:
        benders_data, master_model, master_vars = solve_benders_linearized(
            ext_form=ext_form,
            alpha=0.1,
            max_iterations=max_iterations,
            tolerance=1e-4,
            verbose=verbose,
        )
        solve_time = time.time() - start_time

        objective = benders_data.lower_bound
        x_sol = benders_data.x_current
        n_facilities = sum(1 for v in x_sol.values() if v > 0.5) if x_sol else 0
        gap = benders_data.gap

        print(f"Status: CONVERGED")
        print(f"Objective: {objective:.2f}")
        print(f"Gap: {gap:.4%}")
        print(f"Facilities: {n_facilities}")
        print(f"Iterations: {benders_data.iteration}")
        print(f"Time: {solve_time:.2f}s")

        return {
            'method': 'McCormick + Benders',
            'n_scenarios': n_scenarios,
            'status': 'converged',
            'objective': objective,
            'gap': gap,
            'time': solve_time,
            'iterations': benders_data.iteration,
            'n_facilities': n_facilities,
        }
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"Status: FAILED ({str(e)})")
        return {
            'method': 'McCormick + Benders',
            'n_scenarios': n_scenarios,
            'status': 'failed',
            'objective': None,
            'time': solve_time,
            'error': str(e),
        }


def run_comprehensive_benchmark(
    scenarios_list: List[int] = [10, 50, 100],
    n_price_levels_list: List[int] = [5, 13, 21],
    instance_size: str = "medium",
    seed: int = 42,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Run comprehensive benchmark comparing all three approaches.

    Args:
        scenarios_list: List of scenario counts to test
        n_price_levels_list: List of price discretization levels
        instance_size: "small", "medium", or "large"
        seed: Random seed
        save_results: Save results to CSV
    """

    # Create instance
    if instance_size == "small":
        cfg = Config(n_customers=5, n_facilities=3, seed=seed)
    elif instance_size == "medium":
        cfg = Config(n_customers=20, n_facilities=10, seed=seed)
    else:  # large
        cfg = Config(n_customers=50, n_facilities=20, seed=seed)

    inst = Instance.from_config(cfg)

    print(f"\n{'#'*70}")
    print(f"COMPREHENSIVE BENCHMARK")
    print(f"{'#'*70}")
    print(f"Instance: {len(inst.I)} customers, {len(inst.J)} facilities")
    print(f"Scenarios to test: {scenarios_list}")
    print(f"Discrete price levels: {n_price_levels_list}")

    results = []

    # Test each scenario count
    for n_scenarios in scenarios_list:
        print(f"\n{'*'*70}")
        print(f"Testing with {n_scenarios} scenarios")
        print(f"{'*'*70}")

        # 1. Extensive Form (baseline)
        result = run_extensive_form(inst, n_scenarios, seed=seed, verbose=0)
        results.append(result)
        baseline_obj = result.get('objective')
        baseline_time = result.get('time')

        # 2. McCormick + Benders
        result = run_mccormick_benders(inst, n_scenarios, seed=seed, verbose=0)
        # Add comparison to baseline
        if baseline_obj and result.get('objective'):
            result['gap_vs_optimal'] = (result['objective'] - baseline_obj) / abs(baseline_obj)
        if baseline_time and result.get('time'):
            result['speedup'] = baseline_time / result['time']
        results.append(result)

        # 3. Discrete Prices + Benders (test different price levels)
        for n_price_levels in n_price_levels_list:
            result = run_discrete_price_benders(
                inst, n_scenarios, n_price_levels=n_price_levels,
                seed=seed, verbose=0
            )
            # Add comparison to baseline
            if baseline_obj and result.get('objective'):
                result['gap_vs_optimal'] = (result['objective'] - baseline_obj) / abs(baseline_obj)
            if baseline_time and result.get('time'):
                result['speedup'] = baseline_time / result['time']
            results.append(result)

        # 4. Progressive Hedging
        # We can test a default rho value, or a list of them
        default_rho = 1000.0
        result = run_progressive_hedging(
            inst, n_scenarios, rho=default_rho, seed=seed, max_iterations=100, verbose=0
        )
        # Add comparison to baseline
        if baseline_obj and result.get('objective'):
            result['gap_vs_optimal'] = (result['objective'] - baseline_obj) / abs(baseline_obj)
        if baseline_time and result.get('time'):
            result['speedup'] = baseline_time / result['time']
        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Display summary
    print(f"\n{'#'*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'#'*70}\n")
    print(df.to_string(index=False))

    # Save results
    if save_results:
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        # Create descriptive filename based on parameters
        scenarios_str = "_".join(map(str, scenarios_list))
        prices_str = "_".join(map(str, n_price_levels_list))
        filename_base = f"scenarios_{scenarios_str}_prices_{prices_str}_{instance_size}"

        csv_file = os.path.join("results", f"{filename_base}.csv")
        df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {csv_file}")

        # Also save as JSON for better structure
        json_file = os.path.join("results", f"{filename_base}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {json_file}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SFLPP solution methods"
    )
    parser.add_argument(
        '--scenarios', nargs='+', type=int, default=[10, 50, 100],
        help='List of scenario counts to test (default: 10 50 100)'
    )
    parser.add_argument(
        '--n_prices', nargs='+', type=int, default=[5, 13, 21],
        help='List of discrete price levels (default: 5 13 21)'
    )
    parser.add_argument(
        '--size', choices=['small', 'medium', 'large'], default='medium',
        help='Instance size (default: medium)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Do not save results to file'
    )

    args = parser.parse_args()

    run_comprehensive_benchmark(
        scenarios_list=args.scenarios,
        n_price_levels_list=args.n_prices,
        instance_size=args.size,
        seed=args.seed,
        save_results=not args.no_save,
    )


if __name__ == "__main__":
    main()
