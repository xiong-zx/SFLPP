"""
Benchmark_2: Compare Continuous Price MIQP vs. Discrete Price MILP.

Loads a pre-generated instance and its sampled extensive form scenarios, then:
  1) Solves the extensive form MIQP with continuous prices.
  2) Solves the discrete-price MILP where prices are discretized into levels.

Results are printed and optionally saved to results/ as CSV and JSON.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import gurobipy as gp

from core.data import Instance
from core.extensive_form import ExtensiveForm
from core.extensive_form_fixed_price import build_extensive_form_fixed_price_model
from core.discrete_price_milp import build_discrete_price_milp_model
from core.solver import load_gurobi_params, solve_gurobi_model

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass
class Benchmark2Settings:
    config_name: str = "c5_f5_cf1"
    instance_idx: Optional[List[int]] = None
    scenarios_list: Optional[List[int]] = None
    alpha: float = 0.1
    price_levels: int = 10  # Number of discrete price levels for MILP
    price_levels_list: Optional[List[int]] = None
    save_results: bool = True

    def __post_init__(self):
        if self.instance_idx is None:
            self.instance_idx = [1]
        if self.scenarios_list is None:
            self.scenarios_list = [10, 20, 50]
        if self.price_levels_list is None:
            self.price_levels_list = [self.price_levels]


def instance_file_path(config_name: str, instance_idx: int) -> Path:
    return DATA_DIR / f"{config_name}_ins{instance_idx}.json"


def ef_file_path(config_name: str, instance_idx: int, n_scenarios: int) -> Path:
    return DATA_DIR / f"{config_name}_ins{instance_idx}_s{n_scenarios}.pkl"


# --------------------------------------------------------------------------- #
# Gurobi helper
# --------------------------------------------------------------------------- #
def apply_gurobi_defaults(params: Dict[str, float | str]) -> None:
    """Apply default Gurobi parameters globally so all models inherit them."""
    for key, val in params.items():
        gp.setParam(key, val)


# --------------------------------------------------------------------------- #
# Solution methods
# --------------------------------------------------------------------------- #
def run_extensive_form(
    ext_form: ExtensiveForm,
    alpha: float,
    verbose: int = 0,
    gurobi_params: Dict[str, float | str] | None = None,
) -> Dict:
    """
    Solve the extensive form MIQP with continuous prices.
    """
    n_scenarios = len(ext_form.scenarios)
    start = time.time()
    model, vars_dict = build_extensive_form_fixed_price_model(ext_form, alpha=alpha)
    params = dict(gurobi_params or {})
    params.setdefault("OutputFlag", verbose)
    solve_gurobi_model(model, params=params)
    elapsed = time.time() - start

    status = model.Status
    result: Dict = {
        "method": "Continuous Price MIQP",
        "status": status,
        "objective": None,
        "time": elapsed,
        "n_scenarios": n_scenarios,
    }
    if status == gp.GRB.OPTIMAL:
        result["objective"] = model.ObjVal
        result["n_facilities"] = sum(1 for j in ext_form.J if vars_dict["x"][j].X > 0.5)

        # Print x and p values
        print("\n--- Continuous Price MIQP Solution ---")
        print("x (facility locations):")
        for j in ext_form.J:
            if vars_dict["x"][j].X > 0.5:
                print(f"  Facility {j}: OPEN (x={vars_dict['x'][j].X:.4f})")

        if "p" in vars_dict:
            print("\np (prices per customer):")
            for i in ext_form.I:
                print(f"  Customer {i}: p={vars_dict['p'][i].X:.4f}")

    return result


def run_discrete_price_milp(
    ext_form: ExtensiveForm,
    alpha: float,
    price_levels: int,
    verbose: int = 0,
    gurobi_params: Dict[str, float | str] | None = None,
) -> Dict:
    """
    Solve the discrete price MILP where prices are discretized into levels.
    """
    n_scenarios = len(ext_form.scenarios)
    start = time.time()
    model, vars_dict = build_discrete_price_milp_model(
        ext_form, alpha=alpha, price_levels=price_levels
    )
    params = dict(gurobi_params or {})
    params.setdefault("OutputFlag", verbose)
    solve_gurobi_model(model, params=params)
    elapsed = time.time() - start

    status = model.Status
    result: Dict = {
        "method": "Discrete Price MILP",
        "status": status,
        "objective": None,
        "time": elapsed,
        "n_scenarios": n_scenarios,
        "price_levels": price_levels,
    }
    if status == gp.GRB.OPTIMAL:
        result["objective"] = model.ObjVal
        result["n_facilities"] = sum(1 for j in ext_form.J if vars_dict["x"][j].X > 0.5)

        # Print x and p values
        print("\n--- Discrete Price MILP Solution ---")
        print("x (facility locations):")
        for j in ext_form.J:
            if vars_dict["x"][j].X > 0.5:
                print(f"  Facility {j}: OPEN (x={vars_dict['x'][j].X:.4f})")

        if "p" in vars_dict:
            print("\np (prices per customer):")
            for i in ext_form.I:
                price_val = vars_dict["p"][i].X
                # Find which discrete level was selected
                if "z" in vars_dict:
                    selected_level = None
                    for l in range(price_levels):
                        if vars_dict["z"][i, l].X > 0.5:
                            selected_level = l
                            break
                    print(f"  Customer {i}: p={price_val:.4f} (level {selected_level})")
                else:
                    print(f"  Customer {i}: p={price_val:.4f}")

    return result


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_benchmark(settings: Benchmark2Settings) -> pd.DataFrame:
    """
    Compare Continuous Price MIQP vs Discrete Price MILP across instances and scenario counts.
    """
    config_name = settings.config_name
    instance_list = settings.instance_idx
    scenarios_list = settings.scenarios_list
    price_levels_list = settings.price_levels_list

    gurobi_params = load_gurobi_params(CONFIG_DIR / "gurobi_params.json")
    apply_gurobi_defaults(gurobi_params)

    all_results: List[Dict] = []

    for inst_idx in instance_list:
        inst_path = instance_file_path(config_name, inst_idx)
        if not inst_path.exists():
            raise FileNotFoundError(f"Instance file not found: {inst_path}")

        print(f"\n{'#'*70}")
        print(f"BENCHMARK_2 | config={config_name} | instance={inst_idx}")
        print(f"Scenario counts: {scenarios_list} | alpha={settings.alpha} | price_levels={price_levels_list}")
        print(f"{'#'*70}")

        # Load instance (for info only)
        inst = Instance.load_json(str(inst_path))
        print(f"Instance: {len(inst.I)} customers, {len(inst.J)} facilities")

        for n_scen in scenarios_list:
            ef_path = ef_file_path(config_name, inst_idx, n_scen)
            if not ef_path.exists():
                raise FileNotFoundError(f"Extensive form file not found: {ef_path}")
            ext_form = ExtensiveForm.load_pkl(str(ef_path))

            print(f"\n{'*'*70}")
            print(f"Scenarios = {n_scen}")
            print(f"{'*'*70}")

            # 1) Continuous price MIQP baseline
            ef_res = run_extensive_form(
                ext_form,
                alpha=settings.alpha,
                verbose=0,
                gurobi_params=gurobi_params,
            )
            ef_res["instance"] = inst_idx
            ef_res["config"] = config_name
            all_results.append(ef_res)
            print(f"[Continuous Price MIQP] status={ef_res['status']} obj={ef_res['objective']} time={ef_res['time']:.2f}s")

            # 2) Discrete price MILP (sweep price levels)
            for p_levels in price_levels_list:
                discrete_res = run_discrete_price_milp(
                    ext_form,
                    alpha=settings.alpha,
                    price_levels=p_levels,
                    verbose=0,
                    gurobi_params=gurobi_params,
                )
                discrete_res["instance"] = inst_idx
                discrete_res["config"] = config_name

                if ef_res.get("objective") is not None and discrete_res.get("objective") is not None:
                    continuous_obj = ef_res["objective"]
                    discrete_obj = discrete_res["objective"]
                    discrete_res["gap_vs_continuous"] = (discrete_obj - continuous_obj) / abs(continuous_obj) if abs(continuous_obj) > 1e-8 else None

                all_results.append(discrete_res)
                print(
                    f"[Discrete Price MILP ] levels={p_levels} status={discrete_res['status']} "
                    f"obj={discrete_res['objective']} time={discrete_res['time']:.2f}s "
                    f"gap_vs_continuous={discrete_res.get('gap_vs_continuous')}"
                )

    df = pd.DataFrame(all_results)

    # Summary and save
    print(f"\n{'#'*70}")
    print("BENCHMARK_2 SUMMARY")
    print(f"{'#'*70}\n")
    print(df.to_string(index=False))

    if settings.save_results and len(all_results) > 0:
        Path("results").mkdir(exist_ok=True)
        scenarios_str = "_".join(map(str, scenarios_list))
        price_str = "_".join(map(str, price_levels_list))
        inst_str = "_".join(map(str, instance_list))
        base = f"{config_name}_ins{inst_str}_scenarios_{scenarios_str}_prices_{price_str}_bench2"
        csv_path = Path("results") / f"{base}.csv"
        json_path = Path("results") / f"{base}.json"
        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {csv_path}")
        print(f"Results saved to: {json_path}")

    return df


if __name__ == "__main__":
    SETTINGS = Benchmark2Settings(
        config_name="c5_f5_cf1",
        instance_idx=[1],
        scenarios_list=[10, 20, 50, 100, 200],
        price_levels_list=[3, 5, 8, 10, 15],
        alpha=0.1,
        save_results=True,
    )
    run_benchmark(SETTINGS)
