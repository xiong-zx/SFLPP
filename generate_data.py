"""
Batch generator for SFLPP instances and extensive forms using existing configs.

Naming conventions:
  - Configs:   expected at config/{config_name}.json (already existing)
  - Instance:  {config_name}_ins{ins_idx}.json
  - EF:        {config_name}_ins{ins_idx}_s{scenarios}.pkl
"""

# %%
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterator, Sequence, List

import numpy as np

from core.data import Config, Instance
from core.extensive_form import ExtensiveForm, sample_extensive_form
from core.utils import ef_file_path, instance_file_path, config_file_path

# --- Global Switch ---
# Set to True to use the version with distance-based costs and visualization data.
# Set to False to use the original simple version.
USE_DIST_VERSION = True


@dataclass
class GenerationSettings:
    config_names: Sequence[str] = ()
    instance_idx: int | List[int] = 1
    scenarios: Sequence[int] = ()
    base_seed: int | None = 0  # set to None for nondeterministic seeds
    overwrite: bool = False


def seed_stream(base_seed: int | None) -> Iterator[int]:
    """Infinite stream of integer seeds derived from a base seed."""
    rng = np.random.default_rng(base_seed)
    while True:
        yield int(rng.integers(0, 2**32 - 1))


def instance_name(config_name: str, inst_idx: int) -> str:
    return f"{config_name}_ins{inst_idx}"


def ef_name(config_name: str, inst_idx: int, n_scenarios: int) -> str:
    return f"{instance_name(config_name, inst_idx)}_s{n_scenarios}"


def save_instance(inst: Instance, path: Path, overwrite: bool) -> bool:
    """Save instance if needed; return True if written."""
    if path.exists() and not overwrite:
        print(f"[skip] Instance exists: {path.name}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    inst.save_json(str(path))
    print(f"[write] Instance -> {path.name} (seed={inst.config.seed})")
    return True


def save_extensive_form(ext: ExtensiveForm, path: Path, overwrite: bool) -> bool:
    """Save extensive form if needed; return True if written."""
    if path.exists() and not overwrite:
        print(f"[skip] EF exists: {path.name}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    ext.save_pkl(str(path))
    print(f"[write] EF      -> {path.name}")
    return True


def generate_bundle(
    config_names: Sequence[str],
    instance_idx: int | List[int],
    scenarios: Sequence[int],
    base_seed: int | None,
    overwrite: bool,
) -> Dict[str, int]:
    """
    Generate instances and extensive forms for the provided configs.
    Returns counts of written artifacts.
    """
    if not config_names:
        raise ValueError("No config_names provided. Edit SETTINGS.config_names.")

    stats = {"instances": 0, "efs": 0}
    seeds = seed_stream(base_seed)

    if isinstance(instance_idx, int):
        instance_idx = [instance_idx]

    for config_name in config_names:
        cfg_path = config_file_path(config_name)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        cfg = Config.load_json(str(cfg_path))

        for inst_idx in instance_idx:
            inst_seed = next(seeds)
            inst_cfg = replace(cfg, seed=inst_seed)
            inst = Instance.from_config(inst_cfg, use_distance_costs=USE_DIST_VERSION)
            inst_path = instance_file_path(config_name, inst_idx)

            if save_instance(inst, inst_path, overwrite):
                stats["instances"] += 1

            for n_scenarios in scenarios:
                ef_seed = next(seeds)
                ext = sample_extensive_form(inst, n_scenarios=n_scenarios, seed=ef_seed)
                ef_path = ef_file_path(config_name, inst_idx, n_scenarios)
                if save_extensive_form(ext, ef_path, overwrite):
                    stats["efs"] += 1

    return stats


# %%
if __name__ == "__main__":
    # Edit settings here as needed.
    settings = GenerationSettings(
        config_names=["c20_f10_cf5"],
        instance_idx=[2],
        scenarios=[10, 20, 50, 100, 200],
        base_seed=0,
        overwrite=False,
    )

    stats = generate_bundle(
        config_names=settings.config_names,
        instance_idx=settings.instance_idx,
        scenarios=settings.scenarios,
        base_seed=settings.base_seed,
        overwrite=settings.overwrite,
    )

    print(
        f"\nDone. Wrote {stats['instances']} instances, "
        f"{stats['efs']} extensive forms."
    )
# %%
