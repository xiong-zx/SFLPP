"""
Extensive-form solver runner that loads pre-generated data and runs multiple experiments.
Edit CONFIG_NAME / INSTANCE_IDX_LIST / SCENARIOS_LIST below and run.
Jupyter-friendly: experiments are executed directly under __main__.
"""

# %%
import json
from pathlib import Path
from typing import Dict, List
from time import time

from core.data import Instance
from core.extensive_form import ExtensiveForm, build_extensive_form_model
from core.solver import solve_gurobi_model, load_gurobi_params

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "log"


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

    start_time = time()
    solve_gurobi_model(
        model,
        log_file=LOG_DIR / f"{config_name}_ins{instance_idx}_s{n_scenarios}.log",
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
        "opened_facilities": [
            j for j in inst.J if model.Status == 2 and vars_dict["x"][j].X > 0.5
        ],
    }

    log_name = f"{config_name}_ins{instance_idx}_s{n_scenarios}_log.json"
    log_path = LOG_DIR / log_name
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"Saved log to {log_path.name} | "
        f"status={model.Status}, obj={result['objective']}"
    )

    return result


# %%
if __name__ == "__main__":
    CONFIG_NAME = "c5_f5_cf1"  # corresponds to config/{CONFIG_NAME}.json
    INSTANCE_IDX_LIST: List[int] = [1]
    SCENARIOS_LIST: List[int] = [10, 20, 50]
    gurobi_params = load_gurobi_params(CONFIG_DIR / "gurobi_params.json")
    all_results = []
    for inst_idx in INSTANCE_IDX_LIST:
        for n_scenarios in SCENARIOS_LIST:
            res = run_experiment(CONFIG_NAME, inst_idx, n_scenarios, gurobi_params)
            all_results.append(res)

# %%
