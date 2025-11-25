"""
Generate EF variants with regional tariff rules while keeping capacities fixed.

Rules:
- If customer-facility distance <= LOCAL_RADIUS: tariff = 0.
- Else if they are in different regions/countries: tariff = tau_level (sweep list).
- Else (same region, non-local): tariff = 0.

Capacities u_j^w and scenario weights are reused from a baseline EF pickle so only tariffs change.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
import math
import json

from core.data import Instance
from core.extensive_form import ExtensiveForm
from core.extensive_form_fixed_price import build_extensive_form_fixed_price_model
from core.solver import solve_gurobi_model
from visualize_solution import visualize_network


@dataclass
class SweepSettings:
    config_name: str = "c20_f5_cf1"
    instance_idx: int = 1
    n_scenarios: int = 10
    tau_levels: List[float] = (0.05, 0.2, 0.8, 1.2)
    local_radius: float = 10.0  # distance threshold for zero tariff
    x_boundary: float = 40.0  # split for region A/B by x-coordinate
    y_boundary: float = 40.0  # split for region A/B by y-coordinate
    use_y_boundary: bool = False  # if True, use y_boundary instead of x_boundary
    war_regions: tuple[str, str] = ("A", "B")  # tariff war between these regions
    base_tau_min: float = 0.0  # baseline tariff range for non-war pairs
    base_tau_max: float = 0.1
    # Region preference: make one region cheaper to open to induce cross-border flows initially
    favor_region: str  | None = "A" # None to disable; else "A"/"B"
    favor_region_factor: float = 0.2
    other_region_factor: float = 3
    favor_region_demand_factor: float = 0.3  # scale demand in favored region (b_i & base_demand)
    use_dist_version: bool = True
    baseline_suffix: str | None = None  # None -> default baseline filename
    seed: int | None = 42  # only used to name files; tariffs are deterministic here


def region_from_coord(xy: tuple[float, float], boundary: float, use_y: bool = False) -> str:
    coord = xy[1] if use_y else xy[0]
    return "A" if coord <= boundary else "B"


def build_bar_c(
    inst: Instance,
    base_ext: ExtensiveForm,
    tau_level: float,
    local_radius: float,
    boundary: float,
    war_regions: tuple[str, str],
    base_tau_min: float,
    base_tau_max: float,
    rng,
    use_y_boundary: bool,
):
    I, J = inst.I, inst.J
    c = inst.c
    coords_c = inst.customer_coords
    coords_f = inst.facility_coords
    if not coords_c or not coords_f:
        raise ValueError("Instance must have customer/facility coordinates for regional tariffs.")

    # Precompute region tags
    region_c = {i: region_from_coord(coords_c[i], boundary, use_y=use_y_boundary) for i in I}
    region_f = {j: region_from_coord(coords_f[j], boundary, use_y=use_y_boundary) for j in J}

    import numpy as _np
    rng_local = rng if rng is not None else _np.random.default_rng(0)

    scenarios = []
    for scen in base_ext.scenarios:
        bar_c = {}
        for i in I:
            for j in J:
                dx = coords_c[i][0] - coords_f[j][0]
                dy = coords_c[i][1] - coords_f[j][1]
                dist = math.hypot(dx, dy)
                if dist <= local_radius:
                    tau = 0.0
                elif (region_c[i], region_f[j]) in (war_regions, war_regions[::-1]):
                    tau = tau_level
                else:
                    tau = float(rng_local.uniform(base_tau_min, base_tau_max))
                bar_c[(i, j)] = c[(i, j)] * (1.0 + tau)
        scenarios.append(type(scen)(bar_c=bar_c, u=scen.u, weight=scen.weight))
    return scenarios


def main():
    settings = SweepSettings()
    ROOT = Path(__file__).resolve().parent
    DATA_DIR = ROOT / ("data_dist" if settings.use_dist_version else "data")
    RESULTS_DIR = ROOT / ("results_dist" if settings.use_dist_version else "results")
    PLOTS_DIR = ROOT / ("plots_dist" if settings.use_dist_version else "plots")
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)

    baseline_name = (
        f"{settings.config_name}_ins{settings.instance_idx}_s{settings.n_scenarios}"
        if not settings.baseline_suffix
        else f"{settings.config_name}_ins{settings.instance_idx}_s{settings.n_scenarios}_{settings.baseline_suffix}"
    )
    inst_path = DATA_DIR / f"{settings.config_name}_ins{settings.instance_idx}.json"
    base_ef_path = DATA_DIR / f"{baseline_name}.pkl"

    if not inst_path.exists():
        raise FileNotFoundError(f"Instance not found: {inst_path}")
    if not base_ef_path.exists():
        raise FileNotFoundError(f"Baseline EF not found: {base_ef_path}")

    inst = Instance.load_json(str(inst_path))
    base_ext = ExtensiveForm.load_pkl(str(base_ef_path))
    if len(base_ext.scenarios) != settings.n_scenarios:
        raise ValueError("Baseline EF scenario count mismatch.")

    # Save settings for reproducibility
    settings_out = RESULTS_DIR / f"{settings.config_name}_ins{settings.instance_idx}_s{settings.n_scenarios}_regional_settings.json"
    settings_out.write_text(json.dumps(asdict(settings), indent=2))

    # region tags
    coords_c = inst.customer_coords
    coords_f = inst.facility_coords
    if not coords_c or not coords_f:
        raise ValueError("Instance must have coordinates to apply regional rules.")
    boundary_val = settings.y_boundary if settings.use_y_boundary else settings.x_boundary
    region_c = {i: region_from_coord(coords_c[i], boundary_val, use_y=settings.use_y_boundary) for i in inst.I}
    region_f = {j: region_from_coord(coords_f[j], boundary_val, use_y=settings.use_y_boundary) for j in inst.J}

    # adjust fixed costs to favor one region (cheaper) vs the other (costlier)
    if settings.favor_region:
        new_f = {}
        for j in inst.J:
            factor = settings.favor_region_factor if region_f[j] == settings.favor_region else settings.other_region_factor
            new_f[j] = inst.f[j] * factor
        inst.f = new_f
    else:
        # fallback to proximity-based scaling
        avg_dist_per_fac = {}
        for j in inst.J:
            dists = []
            for i in inst.I:
                dx = coords_c[i][0] - coords_f[j][0]
                dy = coords_c[i][1] - coords_f[j][1]
                dists.append(math.hypot(dx, dy))
            avg_dist_per_fac[j] = sum(dists) / len(dists) if dists else 0.0
        global_mean_dist = sum(avg_dist_per_fac.values()) / len(avg_dist_per_fac) if avg_dist_per_fac else 1.0
        eps = 1e-6
        new_f = {}
        for j in inst.J:
            raw_factor = global_mean_dist / max(avg_dist_per_fac[j], eps)
            new_f[j] = inst.f[j] * raw_factor
        inst.f = new_f

    # scale demand parameters in favored region
    if settings.favor_region and settings.favor_region_demand_factor != 1.0:
        new_b = {}
        new_base_demand = {}
        for i in inst.I:
            if region_c[i] == settings.favor_region:
                scale = settings.favor_region_demand_factor
                new_b[i] = inst.b[i] * scale
                new_base_demand[i] = inst.base_demand[i] * scale
            else:
                new_b[i] = inst.b[i]
                new_base_demand[i] = inst.base_demand[i]
        inst.b = new_b
        inst.base_demand = new_base_demand

    # Gurobi params (optional)
    gurobi_params_path = ROOT / "config" / "gurobi_params.json"
    gurobi_params = json.loads(gurobi_params_path.read_text()) if gurobi_params_path.exists() else None

    sols: List[dict] = []
    exts: List[ExtensiveForm] = []
    global_max_flow = 0.0

    for tau in settings.tau_levels:
        import numpy as np
        rng = np.random.default_rng(settings.seed)
        new_scenarios = build_bar_c(
            inst,
            base_ext,
            tau_level=tau,
            local_radius=settings.local_radius,
            boundary=boundary_val,
            war_regions=settings.war_regions,
            base_tau_min=settings.base_tau_min,
            base_tau_max=settings.base_tau_max,
            rng=rng if rng is not None else None,
            use_y_boundary=settings.use_y_boundary,
        )
        new_ext = ExtensiveForm(instance=inst, scenarios=new_scenarios)
        suffix = f"tau{str(tau).replace('.', 'p')}_regional"
        out_name = f"{settings.config_name}_ins{settings.instance_idx}_s{settings.n_scenarios}_{suffix}.pkl"
        out_path = DATA_DIR / out_name
        new_ext.save_pkl(str(out_path))
        print(f"[write] {out_path.name} (tau={tau})")

        # solve extensive form
        model, vars_dict = build_extensive_form_fixed_price_model(new_ext, alpha=0.1)
        # Ensure nonconvex terms are allowed
        model.setParam("NonConvex", 2)
        solve_gurobi_model(model, params=gurobi_params)
        has_sol = model.SolCount > 0 and model.Status in (2, 11, 12, 13)  # OPTIMAL, SUBOPTIMAL, USER_OBJ_LIMIT, WORK_LIMIT
        sol = {
            "objective": model.ObjVal if has_sol else None,
            "status": model.Status,
            "has_solution": has_sol,
            "x": {j: vars_dict["x"][j].X if has_sol else None for j in new_ext.J},
            "p": {i: vars_dict["p"][i].X if has_sol else None for i in new_ext.I},
            "q": {(i, j, w): vars_dict["q"][i, j, w].X if has_sol else None for i in new_ext.I for j in new_ext.J for w in new_ext.W},
            "s": {(i, w): vars_dict["s"][i, w].X if has_sol else None for i in new_ext.I for w in new_ext.W},
        }
        sols.append({"tau": tau, "sol": sol})
        exts.append(new_ext)
        if has_sol and sol["q"]:
            local_max = max(val for val in sol["q"].values() if val is not None)
            if local_max > global_max_flow:
                global_max_flow = local_max

    # visualize with global flow/tariff scale
    tau_scale_max = max(settings.tau_levels) if settings.tau_levels else None
    for item, ext in zip(sols, exts):
        tau = item["tau"]
        sol = item["sol"]
        if not sol.get("has_solution"):
            print(f"[solve] tau={tau} status={sol['status']} obj={sol['objective']} (no solution)")
            continue
        max_flow_scale = global_max_flow if global_max_flow > 0 else None
        scen_idx = 0
        # build flows and tariffs for scenario scen_idx
        flows = {(i, j): sol["q"].get((i, j, scen_idx), 0.0) for i in ext.I for j in ext.J}
        scen = ext.scenarios[scen_idx]
        tariffs = {}
        for i in ext.I:
            for j in ext.J:
                base_c = ext.instance.c[(i, j)]
                tariffs[(i, j)] = (scen.bar_c[(i, j)] - base_c) / base_c if base_c != 0 else 0.0

        solution_dict = {
            "opened_facilities": [j for j, v in sol["x"].items() if (v is not None and v > 0.5)],
            "flows": flows,
            "prices": sol["p"],
            "served_demand": {i: sol["s"][(i, scen_idx)] for i in ext.I},
            "tariffs": tariffs,
        }
        plot_path = PLOTS_DIR / f"{settings.config_name}_ins{settings.instance_idx}_s{settings.n_scenarios}_tau{str(tau).replace('.', 'p')}_regional.png"
        visualize_network(
            ext.instance,
            solution_dict,
            title=f"{settings.config_name} ins{settings.instance_idx} tau={tau}",
            save_path=plot_path,
            max_flow_scale=max_flow_scale,
            tau_scale_max=tau_scale_max,
            boundary_val=boundary_val,
            use_y_boundary=settings.use_y_boundary,
        )
        print(f"[solve] tau={tau} status={sol['status']} obj={sol['objective']}")


if __name__ == "__main__":
    main()
