"""
Benchmark_2: Compare Extensive Form vs. Benders (discrete prices).

Loads a pre-generated instance and its sampled extensive form scenarios, then:
  1) Solves the extensive form MIQP.
  2) Solves the discrete-price Benders decomposition implemented in
     benders_first_stage_price.py.

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
from benders_first_stage_price import ProblemData, BendersDecomposition

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
    price_levels: int = 3  # kept for backward compatibility (single run)
    price_levels_list: Optional[List[int]] = None
    max_iter: int = 100
    tol: float = 1e-4
    verbose_benders: bool = False
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
# Helpers
# --------------------------------------------------------------------------- #
def build_problem_data_from_ext_form(ext_form: ExtensiveForm, alpha: float) -> ProblemData:
    """
    Convert an ExtensiveForm object into ProblemData for the Benders solver.
    """
    inst = ext_form.instance
    I, J, W = inst.I, inst.J, ext_form.W

    n_customers = len(I)
    n_facilities = len(J)
    n_scenarios = len(W)

    f = np.array([inst.f[j] for j in J])
    c = np.zeros((n_customers, n_facilities))
    a = np.zeros(n_customers)
    b = np.zeros(n_customers)
    for i_idx, i in enumerate(I):
        a[i_idx] = inst.a[i]
        b[i_idx] = inst.b[i]
        for j_idx, j in enumerate(J):
            c[i_idx, j_idx] = inst.c[(i, j)]

    g = np.zeros((n_scenarios, n_customers, n_facilities))
    u = np.zeros((n_scenarios, n_facilities))
    pi = np.zeros(n_scenarios)

    for w_idx, w in enumerate(W):
        scen = ext_form.scenarios[w]
        pi[w_idx] = scen.weight
        for j_idx, j in enumerate(J):
            u[w_idx, j_idx] = scen.u[j]
        for i_idx, i in enumerate(I):
            for j_idx, j in enumerate(J):
                bar_c = scen.bar_c[(i, j)]
                g[w_idx, i_idx, j_idx] = bar_c - inst.c[(i, j)]

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
        pi=pi,
    )


def run_extensive_form(ext_form: ExtensiveForm, alpha: float, verbose: int = 0) -> Dict:
    n_scenarios = len(ext_form.scenarios)
    start = time.time()
    model, vars_dict = build_extensive_form_fixed_price_model(ext_form, alpha=alpha)
    model.setParam("OutputFlag", verbose)
    model.optimize()
    elapsed = time.time() - start

    status = model.Status
    result: Dict = {
        "method": "Extensive Form",
        "status": status,
        "objective": None,
        "time": elapsed,
        "n_scenarios": n_scenarios,
    }
    if status == gp.GRB.OPTIMAL:
        result["objective"] = model.ObjVal
        result["n_facilities"] = sum(1 for j in ext_form.J if vars_dict["x"][j].X > 0.5)
    return result


def run_benders_discrete(ext_form: ExtensiveForm, alpha: float, settings: Benchmark2Settings, price_levels: int) -> Dict:
    pd_data = build_problem_data_from_ext_form(ext_form, alpha=alpha)
    solver = BendersDecomposition(
        pd_data,
        verbose=settings.verbose_benders,
        price_levels=price_levels,
    )

    start = time.time()
    result = solver.solve(max_iter=settings.max_iter, tol=settings.tol)
    elapsed = time.time() - start

    if result is None:
        return {
            "method": "Benders (discrete price)",
            "status": "failed",
            "objective": None,
            "time": elapsed,
            "gap": None,
            "iterations": solver.iteration + 1,
            "n_scenarios": len(ext_form.scenarios),
        }

    # Re-evaluate using served flows to make objective comparable to extensive form
    eval_result = solver.evaluate_solution(result["x"], result["p"])
    obj_true = eval_result.get("total_cost")
    n_facilities = int(np.sum(result["x"] > 0.5))
    return {
        "method": "Benders (discrete price)",
        "status": "ok",
        "objective": obj_true,
        "gap": result["gap"],
        "iterations": result["iterations"],
        "time": elapsed,
        "n_facilities": n_facilities,
        "n_scenarios": len(ext_form.scenarios),
        "price_levels": price_levels,
        "expected_recourse": eval_result.get("expected_recourse"),
        "expected_revenue": eval_result.get("expected_revenue"),
    }


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_benchmark(settings: Benchmark2Settings) -> pd.DataFrame:
    """
    Compare Extensive Form vs Benders (discrete price) across instances and scenario counts.
    """
    config_name = settings.config_name
    instance_list = settings.instance_idx
    scenarios_list = settings.scenarios_list
    price_levels_list = settings.price_levels_list

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

            # 1) Extensive form baseline
            ef_res = run_extensive_form(ext_form, alpha=settings.alpha, verbose=0)
            ef_res["instance"] = inst_idx
            ef_res["config"] = config_name
            all_results.append(ef_res)
            print(f"[Extensive] status={ef_res['status']} obj={ef_res['objective']} time={ef_res['time']:.2f}s")

            # 2) Benders with discrete prices (sweep price levels)
            for p_levels in price_levels_list:
                benders_res = run_benders_discrete(
                    ext_form, alpha=settings.alpha, settings=settings, price_levels=p_levels
                )
                benders_res["instance"] = inst_idx
                benders_res["config"] = config_name
                benders_res["price_levels"] = p_levels

                if ef_res.get("objective") is not None and benders_res.get("objective") is not None:
                    opt = ef_res["objective"]
                    bnd = benders_res["objective"]
                    benders_res["gap_vs_extensive"] = (bnd - opt) / abs(opt) if abs(opt) > 1e-8 else None

                all_results.append(benders_res)
                print(
                    f"[Benders ] levels={p_levels} status={benders_res['status']} obj={benders_res['objective']} "
                    f"gap={benders_res.get('gap')} time={benders_res['time']:.2f}s "
                    f"iters={benders_res.get('iterations')} "
                    f"recourse={benders_res.get('expected_recourse')} "
                    f"revenue={benders_res.get('expected_revenue')}"
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
        scenarios_list=[10, 20, 50],
        price_levels_list=[5, 10, 20, 50, 100],
        alpha=0.1,
        max_iter=50,
        tol=1e-4,
        verbose_benders=False,
        save_results=True,
    )
    run_benchmark(SETTINGS)
