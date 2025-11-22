"""
Progressive Hedging (PH) solver runner that loads pre-generated data and runs multiple experiments.
Edit CONFIG_NAME / INSTANCE_IDX_LIST / SCENARIOS_LIST below and run.
"""

# %%
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple
from time import time

from core.data import Instance
from core.extensive_form import ExtensiveForm
from core.progressive_hedging import solve_with_ph, evaluate_first_stage_solution

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PH_LOG_DIR = ROOT / "ph_log"  # Separate log directory for PH runs
EF_LOG_DIR = ROOT / "results"  # Directory where EF_runner.py saves its results
RESULTS_DIR.mkdir(exist_ok=True)
PH_LOG_DIR.mkdir(exist_ok=True)
MAX_WORKERS = 6


# %%
def run_ph_experiment(
    config_name: str,
    instance_idx: int,
    n_scenarios: int,
    ph_params: Dict,
    optimal_obj: float | None = None, # New parameter to pass the optimal objective
) -> Dict:
    """
    Runs a single Progressive Hedging experiment for a given configuration.
    """
    inst_path = DATA_DIR / f"{config_name}_ins{instance_idx}.json"
    ef_path = DATA_DIR / f"{config_name}_ins{instance_idx}_s{n_scenarios}.pkl"

    if not inst_path.exists():
        raise FileNotFoundError(f"Missing instance file: {inst_path}")
    if not ef_path.exists():
        raise FileNotFoundError(f"Missing extensive form file: {ef_path}")

    inst = Instance.load_json(str(inst_path))
    ext_form = ExtensiveForm.load_pkl(str(ef_path))

    start_time = time()
    ph_results = solve_with_ph(
        ext_form=ext_form,
        rho=ph_params.get("rho", 1000.0),
        max_iter=ph_params.get("max_iter", 100),
        alpha=ph_params.get("alpha", 0.1),
        tol=ph_params.get("tol", 1e-4),
    )
    solve_time = time() - start_time

    # Evaluate the final solution to get the true objective value
    final_x = ph_results["final_x"]
    objective = evaluate_first_stage_solution(final_x, ext_form, alpha=ph_params.get("alpha", 0.1))
    n_facilities = sum(1 for v in final_x.values() if v > 0.5)

    result = {
        "config": config_name,
        "instance_idx": instance_idx,
        "n_scenarios": n_scenarios,
        "ph_params": ph_params,
        "status": "converged" if ph_results["converged"] else "max_iter",
        "solve_time": solve_time,
        "objective": objective,
        "iterations": ph_results["iterations"],
        "opened_facilities": [j for j, v in final_x.items() if v > 0.5],
        "n_facilities": n_facilities,
    }

    # Calculate gap if optimal objective is available
    if optimal_obj is not None and objective is not None:
        if abs(optimal_obj) > 1e-8:
            result["gap_vs_optimal"] = (objective - optimal_obj) / abs(optimal_obj)
        else:
            result["gap_vs_optimal"] = 0.0

    # Save individual log
    json_log_name = f"{config_name}_ins{instance_idx}_s{n_scenarios}_ph_log.json"
    json_log = PH_LOG_DIR / json_log_name
    with open(json_log, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(
        f"Saved log to {json_log.name} | "
        f"status={result['status']}, obj={result['objective']:.2f}"
    )

    return result


def load_optimal_results(summary_path: Path) -> Dict[Tuple[str, int, int], float]:
    """
    Loads optimal results from EF_runner's summary.json into a lookup dictionary.
    """
    if not summary_path.exists():
        print(f"Warning: Optimal results file not found at {summary_path}. Gap will not be calculated.")
        return {}

    with open(summary_path, "r", encoding="utf-8") as f:
        ef_results = json.load(f)

    lookup = {}
    for res in ef_results:
        key = (res["config"], res["instance_idx"], res["n_scenarios"])
        if res.get("objective") is not None:
            lookup[key] = res["objective"]
    return lookup
# %%
if __name__ == "__main__":
    # --- Define experiments to run ---
    CONFIG_NAMES: List[str] = ["c5_f5_cf1", "c5_f10_cf1", "c10_f5_cf1", "c10_f10_cf1",]
    INSTANCE_IDX_LIST: List[int] = [1, 2, 3]
    SCENARIOS_LIST: List[int] = [10, 20, 50]

    # --- PH Algorithm Parameters ---
    ph_params = {"rho": 5000.0, "max_iter": 100, "alpha": 0.1, "tol": 1e-4}

    # --- Load optimal results for gap calculation ---
    optimal_results_lookup = load_optimal_results(EF_LOG_DIR / "summary.json")

    tasks: List[Tuple[str, int, int]] = [
        (cfg, inst_idx, n_scenarios)
        for cfg in CONFIG_NAMES
        for inst_idx in INSTANCE_IDX_LIST
        for n_scenarios in SCENARIOS_LIST
    ]

    all_results: List[Dict] = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(
                run_ph_experiment, cfg, inst_idx, n_scenarios, ph_params,
                optimal_obj=optimal_results_lookup.get((cfg, inst_idx, n_scenarios))
            ): (cfg, inst_idx, n_scenarios)
            for cfg, inst_idx, n_scenarios in tasks
        }
        for future in as_completed(future_to_task):
            task_info = future_to_task[future]
            try:
                res = future.result()
                all_results.append(res)
            except Exception as e:
                cfg, inst_idx, scen = task_info
                print(f"Task failed for {cfg}, ins={inst_idx}, scen={scen}: {e}")

    summary_path = RESULTS_DIR / "ph_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)
    print(f"\nSaved summary of {len(all_results)} runs to {summary_path}")