# %%
from core.data import Config, Instance
from core.extensive_form import sample_extensive_form, build_extensive_form_model
from core.solver import solve_gurobi_model


# %%
def main():
    cfg = Config(
        n_customers=10,
        n_facilities=5,
        p_min=20.0,
        p_max=80.0,
        gamma=0.3,
        seed=42,
    )

    inst = Instance.from_config(cfg)

    ext = sample_extensive_form(inst, n_scenarios=10, seed=123)

    ext.save_json("log/extensive_form_example.json")

    model, vars_dict = build_extensive_form_model(ext, risk_measure="expectation")

    solve_gurobi_model(model, params={"OutputFlag": 1})

    x = vars_dict["x"]
    q = vars_dict["q"]

    if model.Status == 2:  # GRB.OPTIMAL
        print("\n=== Solution summary ===")
        print("Objective:", model.ObjVal)

        open_facilities = [j for j in inst.J if x[j].X > 0.5]
        print("Opened facilities:", open_facilities)

        # Show served quantity for scenario 0
        print("\nServed quantities in scenario 0:")
        w0 = 0
        for i in inst.I:
            total_i = sum(q[i, j, w0].X for j in inst.J)
            if total_i > 1e-6:
                print(f"  Customer {i}: total served = {total_i:.2f}")


# %%
if __name__ == "__main__":
    main()
