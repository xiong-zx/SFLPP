"""
Generate missing extensive form files for specific scenarios.
"""
from pathlib import Path
from core.data import Instance
from core.extensive_form import sample_extensive_form

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

def generate_ef(config_name: str, instance_idx: int, n_scenarios: int, seed: int = None):
    """Generate extensive form file for a specific instance and scenario count."""

    # Load the instance
    inst_path = DATA_DIR / f"{config_name}_ins{instance_idx}.json"
    if not inst_path.exists():
        raise FileNotFoundError(f"Instance file not found: {inst_path}")

    print(f"Loading instance from {inst_path}")
    inst = Instance.load_json(str(inst_path))

    # Generate extensive form
    print(f"Generating extensive form with {n_scenarios} scenarios...")
    ef_seed = seed if seed is not None else hash(f"{config_name}_{instance_idx}_{n_scenarios}") % (2**32)
    ext = sample_extensive_form(inst, n_scenarios=n_scenarios, seed=ef_seed)

    # Save extensive form
    ef_path = DATA_DIR / f"{config_name}_ins{instance_idx}_s{n_scenarios}.pkl"
    ext.save_pkl(str(ef_path))
    print(f"Saved extensive form to {ef_path}")

    return ef_path

if __name__ == "__main__":
    # Generate missing files for c5_f5_cf1_ins1
    config_name = "c5_f5_cf1"
    instance_idx = 1

    # Generate s=100
    print("=" * 70)
    print(f"Generating {config_name}_ins{instance_idx}_s100.pkl")
    print("=" * 70)
    generate_ef(config_name, instance_idx, n_scenarios=100, seed=42)

    # Generate s=200
    print("\n" + "=" * 70)
    print(f"Generating {config_name}_ins{instance_idx}_s200.pkl")
    print("=" * 70)
    generate_ef(config_name, instance_idx, n_scenarios=200, seed=43)

    print("\n" + "=" * 70)
    print("Done! All files generated successfully.")
    print("=" * 70)
