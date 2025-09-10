# run_inference.py
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_sim(sim_path: str):
  spec = importlib.util.spec_from_file_location("sim_mod", sim_path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod

def run_one(sim_mod, data_csv: Path, model_path: str, policy_path: str, device: str, save_traj_dir: Path = None, plot: bool = False):
  # lazy import to avoid torch dependency unless needed
  from controllers.ppo import Controller

  # controller in eval mode (deterministic)
  ctrl = Controller(policy_path=policy_path, training=False, device=device)

  # build sim
  model = sim_mod.TinyPhysicsModel(model_path, debug=False)
  sim   = sim_mod.TinyPhysicsSimulator(model, str(data_csv), controller=ctrl, debug=False)

  # rollout
  cost = sim.rollout()

  # pack a trajectory dataframe (useful for analysis)
  steps = np.arange(len(sim.current_lataccel_history))
  df = pd.DataFrame({
    "step": steps,
    "target_lataccel": sim.target_lataccel_history,
    "current_lataccel": sim.current_lataccel_history,
    "action": sim.action_history + [np.nan]*(len(steps)-len(sim.action_history)),
    "roll_lataccel": [s.roll_lataccel for s in sim.state_history],
    "v_ego": [s.v_ego for s in sim.state_history],
    "a_ego": [s.a_ego for s in sim.state_history],
  })

  # optional save
  if save_traj_dir:
    save_traj_dir.mkdir(parents=True, exist_ok=True)
    out_csv = save_traj_dir / (data_csv.stem + "_traj.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved trajectory -> {out_csv}")

  # optional quick plot
  if plot:
    CONTROL_START_IDX = sim_mod.CONTROL_START_IDX
    _, ax = plt.subplots(3, figsize=(12, 8), constrained_layout=True)
    ax[0].plot(df["step"], df["target_lataccel"], label="target_lataccel")
    ax[0].plot(df["step"], df["current_lataccel"], label="current_lataccel")
    ax[0].axvline(CONTROL_START_IDX, color="k", ls="--", alpha=0.5)
    ax[0].set_title("Lateral Acceleration"); ax[0].legend()

    ax[1].plot(df["step"], df["action"], label="action")
    ax[1].axvline(CONTROL_START_IDX, color="k", ls="--", alpha=0.5)
    ax[1].set_title("Action (steer)"); ax[1].legend()

    ax[2].plot(df["step"], df["v_ego"], label="v_ego")
    ax[2].set_title("v_ego"); ax[2].legend()

    plt.show()

  return cost, df

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--sim_module", required=True, help="Path to your simulator .py (defines TinyPhysicsModel/Simulator)")
  p.add_argument("--model_path", required=True, help="Path to the ONNX model used by TinyPhysicsModel")
  p.add_argument("--data_path", required=True, help="CSV file or directory of CSVs")
  p.add_argument("--policy_path", default=str(Path("controllers") / "ppo_policy.pt"), help="Trained PPO checkpoint")
  p.add_argument("--device", default="cpu")
  p.add_argument("--save_traj_dir", default=None, help="Directory to save per-episode trajectories as CSV")
  p.add_argument("--save_summary_csv", default=None, help="Path to save a summary CSV of costs over files")
  p.add_argument("--num_segs", type=int, default=None, help="If data_path is a dir, cap number of files")
  p.add_argument("--plot", action="store_true", help="Show quick plots per-file")
  args = p.parse_args()

  sim_mod = load_sim(args.sim_module)
  data_path = Path(args.data_path)

  rows = []
  if data_path.is_file():
    cost, _ = run_one(sim_mod, data_path, args.model_path, args.policy_path, args.device,
                      save_traj_dir=Path(args.save_traj_dir) if args.save_traj_dir else None,
                      plot=args.plot)
    rows.append({"file": data_path.name, **cost})
  else:
    files = sorted([p for p in data_path.iterdir() if p.suffix.lower()==".csv"])
    if args.num_segs is not None:
      files = files[:args.num_segs]
    for f in files:
      print(f"\n== Inference on: {f.name}")
      cost, _ = run_one(sim_mod, f, args.model_path, args.policy_path, args.device,
                        save_traj_dir=Path(args.save_traj_dir) if args.save_traj_dir else None,
                        plot=args.plot)
      rows.append({"file": f.name, **cost})

  summary = pd.DataFrame(rows)
  print("\nPer-file costs:")
  with pd.option_context("display.max_rows", None, "display.width", 120):
    print(summary)

  if len(summary) > 1:
    print("\nAverages over files:")
    print(summary[["lataccel_cost","jerk_cost","total_cost"]].mean().to_string())

  if args.save_summary_csv:
    Path(args.save_summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.save_summary_csv, index=False)
    print(f"\nSaved summary -> {args.save_summary_csv}")

if __name__ == "__main__":
  main()