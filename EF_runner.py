"""
Extensive-form solver runner that loads pre-generated data and runs multiple experiments.
Edit CONFIG_NAME / INSTANCE_IDX_LIST / SCENARIOS_LIST below and run.
Jupyter-friendly: experiments are executed directly under __main__.
"""

# %%
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple
from time import time

from core.data import Instance
from core.extensive_form import ExtensiveForm, build_extensive_form_model
from core.solver import solve_gurobi_model, load_gurobi_params

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "log"
LOG_DIR.mkdir(exist_ok=True)
MAX_WORKERS = 6


# %%
def run_experiment(
    config_name: str,
    instance_idx: int,
    n_scenarios: int,
    gurobi_params: Dict[str, float | str],
) -> Dict:
    inst_path = DATA_DIR / f"{config_name}_ins{instance_idx}.json"
    ef_path = DATA_DIR / f"{config_name}_ins{instance_idx}_s{n_scenarios}.pkl"

    if not inst_path.exists():
        raise FileNotFoundError(f"Missing instance file: {inst_path}")
    if not ef_path.exists():
        raise FileNotFoundError(f"Missing extensive form file: {ef_path}")

    inst = Instance.load_json(str(inst_path))
    ext = ExtensiveForm.load_pkl(str(ef_path))

    model, vars_dict = build_extensive_form_model(ext, risk_measure="expectation")
    gurobi_log = LOG_DIR / f"{config_name}_ins{instance_idx}_s{n_scenarios}.log"
    if gurobi_log.exists():
        gurobi_log.unlink()

    start_time = time()
    solve_gurobi_model(
        model,
        log_file=gurobi_log,
        params=gurobi_params,
    )
    end_time = time()

    result = {
        "config": config_name,
        "instance_idx": instance_idx,
        "n_scenarios": n_scenarios,
        "gurobi_params": gurobi_params,
        "status": model.Status,
        "solve_time": end_time - start_time,
        "objective": model.ObjVal if model.SolCount else None,
        "opened_facilities": [j for j in inst.J if vars_dict["x"][j].X > 0.5],
        "gap": model.MIPGap,
    }

    json_log_name = f"{config_name}_ins{instance_idx}_s{n_scenarios}_log.json"
    json_log = LOG_DIR / json_log_name
    if json_log.exists():
        json_log.unlink()
    with open(json_log, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(
        f"Saved log to {json_log.name} | "
        f"status={model.Status}, obj={result['objective']}"
    )

    return result


# %%
if __name__ == "__main__":
    CONFIG_NAMES: List[str] = [
        "c5_f5_cf1",
        "c5_f10_cf1",
        "c10_f5_cf1",
        "c10_f10_cf1",
    ]  # list of config basenames (without .json)
    INSTANCE_IDX_LIST: List[int] = [1, 2, 3]
    SCENARIOS_LIST: List[int] = [10, 20, 50]

    gurobi_params = load_gurobi_params(CONFIG_DIR / "gurobi_params.json")

    tasks: List[Tuple[str, int, int]] = []
    for cfg in CONFIG_NAMES:
        for inst_idx in INSTANCE_IDX_LIST:
            for n_scenarios in SCENARIOS_LIST:
                tasks.append((cfg, inst_idx, n_scenarios))

    all_results: List[Dict] = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(
                run_experiment, cfg, inst_idx, n_scenarios, gurobi_params
            ): (
                cfg,
                inst_idx,
                n_scenarios,
            )
            for cfg, inst_idx, n_scenarios in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                res = future.result()
                all_results.append(res)
            except Exception as e:
                cfg, inst_idx, scen = task
                print(f"Task failed for {cfg}, ins={inst_idx}, scen={scen}: {e}")

    summary_path = LOG_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)
    print(f"\nSaved summary of {len(all_results)} runs to {summary_path.name}")

# %%
