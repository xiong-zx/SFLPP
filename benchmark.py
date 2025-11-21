"""
Comprehensive Benchmark: Comparing Solution Methods for SFLPP (pre-generated data)

Three methods:
1. Extensive Form (exact MIQP)
2. McCormick Linearization + Benders (LP relaxation, ~30-44% gap)
3. Discrete Pricing + Benders (adjustable price levels)
4. Progressive Hedging

Config is loaded from config/{CONFIG_NAME}.json.
Data is loaded from pre-generated instance or EF files in `data/`.
Edit SETTINGS below before run.
"""

# %%
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import pandas as pd

from core.data import Instance
from core.extensive_form import ExtensiveForm, build_extensive_form_model
from core.discrete_price import DiscretePriceConfig, solve_benders_discrete_price
from core.benders_linear import solve_benders_linearized
from core.progressive_hedging import evaluate_first_stage_solution, solve_with_ph

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"


# %% Parameters
@dataclass
class BenchmarkSettings:
    config_name: str = "c10_f10_cf1"  # base config filename without .json
    instance_idx: List[int] = None
    scenarios_list: List[int] = None
    n_price_levels_list: List[int] = None
    seed: int = 42  # used for any internal randomness (price configs, etc.)
    save_results: bool = True

    def __post_init__(self):
        if self.instance_idx is None:
            self.instance_idx = [1]
        if self.scenarios_list is None:
            self.scenarios_list = [10, 20, 50]
        if self.n_price_levels_list is None:
            self.n_price_levels_list = [5, 10, 20]


def instance_file_path(config_name: str, instance_idx: int) -> Path:
    return DATA_DIR / f"{config_name}_ins{instance_idx}.json"


def ef_file_path(config_name: str, instance_idx: int, n_scenarios: int) -> Path:
    return DATA_DIR / f"{config_name}_ins{instance_idx}_s{n_scenarios}.pkl"


def run_extensive_form(ext_form: ExtensiveForm, verbose: int = 0) -> Dict:
    """Run extensive form approach."""
    n_scenarios = len(ext_form.scenarios)
    inst = ext_form.instance

    print(f"\n{'='*70}")
    print(f"EXTENSIVE FORM: {n_scenarios} scenarios")
    print(f"{'='*70}")

    start_time = time.time()
    model, vars_dict = build_extensive_form_model(ext_form, alpha=0.1)
    model.setParam("OutputFlag", verbose)
    model.optimize()
    solve_time = time.time() - start_time

    if model.Status == 2:  # Optimal
        objective = model.ObjVal
        x_sol = {j: vars_dict["x"][j].X for j in inst.J}
        n_facilities = sum(1 for v in x_sol.values() if v > 0.5)

        print(f"Status: OPTIMAL")
        print(f"Objective: {objective:.2f}")
        print(f"Facilities: {n_facilities}")
        print(f"Time: {solve_time:.2f}s")

        return {
            "method": "Extensive Form",
            "n_scenarios": n_scenarios,
            "status": "optimal",
            "objective": objective,
            "time": solve_time,
            "n_facilities": n_facilities,
        }
    else:
        print(f"Status: FAILED (status {model.Status})")
        return {
            "method": "Extensive Form",
            "n_scenarios": n_scenarios,
            "status": "failed",
            "objective": None,
            "time": solve_time,
        }


def run_discrete_price_benders(
    ext_form: ExtensiveForm,
    n_price_levels: int = 13,
    seed: int = 42,
    max_iterations: int = 100,
    verbose: int = 0,
) -> Dict:
    """Run discrete pricing + Benders approach."""
    n_scenarios = len(ext_form.scenarios)

    print(f"\n{'='*70}")
    print(
        f"DISCRETE PRICES + BENDERS: {n_scenarios} scenarios, {n_price_levels} price levels"
    )
    print(f"{'='*70}")

    price_config = DiscretePriceConfig(
        n_price_levels=n_price_levels, price_spacing="uniform"
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
            "method": "Discrete Prices + Benders",
            "n_scenarios": n_scenarios,
            "n_price_levels": n_price_levels,
            "status": "converged",
            "objective": objective,
            "gap": gap,
            "time": solve_time,
            "iterations": benders_data.iteration,
            "n_facilities": n_facilities,
        }
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"Status: FAILED ({str(e)})")
        return {
            "method": "Discrete Prices + Benders",
            "n_scenarios": n_scenarios,
            "n_price_levels": n_price_levels,
            "status": "failed",
            "objective": None,
            "time": solve_time,
            "error": str(e),
        }


def run_progressive_hedging(
    ext_form: ExtensiveForm,
    rho: float = 1000.0,
    max_iterations: int = 100,
    verbose: int = 0,
) -> Dict:
    """Run Progressive Hedging approach."""
    n_scenarios = len(ext_form.scenarios)

    print(f"\n{'='*70}")
    print(f"PROGRESSIVE HEDGING: {n_scenarios} scenarios, rho={rho}")
    print(f"{'='*70}")

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

        final_x = ph_results["final_x"]
        objective = evaluate_first_stage_solution(final_x, ext_form, alpha=0.1)
        n_facilities = sum(1 for v in final_x.values() if v > 0.5)

        print(f"Status: {'CONVERGED' if ph_results['converged'] else 'MAX_ITER'}")
        print(f"Objective: {objective:.2f}")
        print(f"Facilities: {n_facilities}")
        print(f"Iterations: {ph_results['iterations']}")
        print(f"Time: {solve_time:.2f}s")

        return {
            "method": "Progressive Hedging",
            "n_scenarios": n_scenarios,
            "rho": rho,
            "status": "converged" if ph_results["converged"] else "max_iter",
            "objective": objective,
            "time": solve_time,
            "iterations": ph_results["iterations"],
            "n_facilities": n_facilities,
        }
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"Status: FAILED ({str(e)})")
        return {
            "method": "Progressive Hedging",
            "n_scenarios": n_scenarios,
            "rho": rho,
            "status": "failed",
            "objective": None,
            "time": solve_time,
            "error": str(e),
        }


def run_mccormick_benders(
    ext_form: ExtensiveForm,
    max_iterations: int = 100,
    verbose: int = 0,
) -> Dict:
    """Run McCormick linearization + Benders approach."""
    n_scenarios = len(ext_form.scenarios)

    print(f"\n{'='*70}")
    print(f"MCCORMICK + BENDERS: {n_scenarios} scenarios")
    print(f"⚠️  WARNING: McCormick creates ~30-44% relaxation gap")
    print(f"{'='*70}")

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
            "method": "McCormick + Benders",
            "n_scenarios": n_scenarios,
            "status": "converged",
            "objective": objective,
            "gap": gap,
            "time": solve_time,
            "iterations": benders_data.iteration,
            "n_facilities": n_facilities,
        }
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"Status: FAILED ({str(e)})")
        return {
            "method": "McCormick + Benders",
            "n_scenarios": n_scenarios,
            "status": "failed",
            "objective": None,
            "time": solve_time,
            "error": str(e),
        }


def run_comprehensive_benchmark(settings: BenchmarkSettings) -> pd.DataFrame:
    """
    Run comprehensive benchmark comparing all three approaches using pre-generated data.
    """
    config_name = settings.config_name
    instance_list = settings.instance_idx
    scenarios_list = settings.scenarios_list
    n_price_levels_list = settings.n_price_levels_list
    seed = settings.seed
    save_results = settings.save_results

    for instance_idx in instance_list:
        inst_path = instance_file_path(config_name, instance_idx)
        if not inst_path.exists():
            raise FileNotFoundError(f"Instance file not found: {inst_path}")
        inst = Instance.load_json(str(inst_path))

        print(f"\n{'#'*70}")
        print(
            f"COMPREHENSIVE BENCHMARK (config={config_name}, instance={instance_idx})"
        )
        print(f"{'#'*70}")
        print(f"Instance: {len(inst.I)} customers, {len(inst.J)} facilities")
        print(f"Scenarios to test: {scenarios_list}")
        print(f"Discrete price levels: {n_price_levels_list}")

        results = []

        # Test each scenario count
        for n_scenarios in scenarios_list:
            ef_path = ef_file_path(config_name, instance_idx, n_scenarios)
            if not ef_path.exists():
                raise FileNotFoundError(f"Extensive form file not found: {ef_path}")
            ext_form = ExtensiveForm.load_pkl(str(ef_path))

            print(f"\n{'*'*70}")
            print(f"Testing with {n_scenarios} scenarios")
            print(f"{'*'*70}")

            # 1. Extensive Form (baseline)
            result = run_extensive_form(ext_form, verbose=0)
            results.append(result)
            baseline_obj = result.get("objective")
            baseline_time = result.get("time")

            # 2. McCormick + Benders
            result = run_mccormick_benders(ext_form, verbose=0)
            # Add comparison to baseline
            if baseline_obj and result.get("objective"):
                result["gap_vs_optimal"] = (result["objective"] - baseline_obj) / abs(
                    baseline_obj
                )
            if baseline_time and result.get("time"):
                result["speedup"] = baseline_time / result["time"]
            results.append(result)

            # 3. Discrete Prices + Benders (test different price levels)
            for n_price_levels in n_price_levels_list:
                result = run_discrete_price_benders(
                    ext_form, n_price_levels=n_price_levels, seed=seed, verbose=0
                )
                # Add comparison to baseline
                if baseline_obj and result.get("objective"):
                    result["gap_vs_optimal"] = (
                        result["objective"] - baseline_obj
                    ) / abs(baseline_obj)
                if baseline_time and result.get("time"):
                    result["speedup"] = baseline_time / result["time"]
                results.append(result)

            # 4. Progressive Hedging
            # We can test a default rho value, or a list of them
            default_rho = 1000.0
            result = run_progressive_hedging(
                ext_form, rho=default_rho, max_iterations=100, verbose=0
            )
            # Add comparison to baseline
            if baseline_obj and result.get("objective"):
                result["gap_vs_optimal"] = (result["objective"] - baseline_obj) / abs(
                    baseline_obj
                )
            if baseline_time and result.get("time"):
                result["speedup"] = baseline_time / result["time"]
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
        filename_base = f"{config_name}_ins{instance_idx}_scenarios_{scenarios_str}_prices_{prices_str}"

        csv_file = os.path.join("results", f"{filename_base}.csv")
        df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {csv_file}")

        # Also save as JSON for better structure
        json_file = os.path.join("results", f"{filename_base}.json")
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {json_file}")

    return df


# %%
if __name__ == "__main__":
    SETTINGS = BenchmarkSettings(
        config_name="c5_f5_cf1",
        instance_idx=[1, 2, 3],
        scenarios_list=[10, 20],
        n_price_levels_list=[5, 13, 21],
        seed=42,
        save_results=True,
    )
    run_comprehensive_benchmark(SETTINGS)
# %%
