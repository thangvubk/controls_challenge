import argparse, importlib.util, os
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

# ---- import your simulator module dynamically ----
def load_sim(sim_path: str):
  spec = importlib.util.spec_from_file_location("sim_mod", sim_path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod

def compute_step_reward(sim, del_t: float, lat_mult: float):
  """
  Reward = negative incremental cost for the newest step.
  Uses the same ingredients as TinyPhysicsSimulator.compute_cost, but per-step.
  """
  t_pred = sim.current_lataccel_history[-1]
  t_targ = sim.target_lataccel_history[-1]
  lat_cost = ((t_targ - t_pred)**2) * 100.0 * lat_mult
  if len(sim.current_lataccel_history) >= 2:
    jerk = (sim.current_lataccel_history[-1] - sim.current_lataccel_history[-2]) / del_t
    jerk_cost = (jerk**2) * 100.0
  else:
    jerk_cost = 0.0
  return -(lat_cost)

def collect_episode(sim_mod, data_path: Path, model_path: str, ppo_ctrl, device):
  # sim setup
  model = sim_mod.TinyPhysicsModel(model_path, debug=False)
  sim   = sim_mod.TinyPhysicsSimulator(model, str(data_path), controller=ppo_ctrl, debug=False)
  ppo_ctrl.reset()

  rewards = []
  CONTROL_START_IDX = sim_mod.CONTROL_START_IDX
  COST_END_IDX      = sim_mod.COST_END_IDX
  DEL_T             = sim_mod.DEL_T
  LAT_MULT          = sim_mod.LAT_ACCEL_COST_MULTIPLIER

  # step manually (not sim.rollout) to get per-step rewards
  total_steps = len(sim.data)
  while sim.step_idx < min(total_steps, COST_END_IDX):
    sim.step()
    # reward only counted once we start controlling & before cost horizon ends
    if sim.step_idx-1 >= CONTROL_START_IDX and sim.step_idx-1 < COST_END_IDX:
      r = compute_step_reward(sim, DEL_T, LAT_MULT)
      rewards.append(float(r))
    else:
      rewards.append(0.0)

  # gather trajectories from controller
  obs, act, logp = ppo_ctrl.pop_trajectory()
  # truncate buffers to rewards length (defensive)
  S = CONTROL_START_IDX
  E = min(len(rewards), len(obs))
  obs, act, logp, rewards = obs[S:E], act[S:E], logp[S:E], np.asarray(rewards[S:E], dtype=np.float32)

  # bootstrap value = 0 (terminal by design)
  values = ppo_ctrl.value_batch(obs).squeeze(-1)
  values = values
  dones = np.zeros(obs.shape[0], dtype=np.float32)
  dones[-1] = 1.0  # episode end

  return obs, act, logp.reshape(-1,1), rewards.reshape(-1,1), values.reshape(-1,1), dones.reshape(-1,1)

def compute_gae(rew, val, done, gamma=0.99, lam=0.95):
  T = len(rew)
  adv = np.zeros((T,1), dtype=np.float32)
  lastgaelam = 0.0
  next_value = 0.0
  for t in reversed(range(T)):
    nonterminal = 1.0 - done[t,0]
    delta = rew[t,0] + gamma * next_value * nonterminal - val[t,0]
    lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    adv[t,0] = lastgaelam
    next_value = val[t,0]
  ret = adv + val
  # normalize advantages
  adv = (adv - adv.mean()) / (adv.std() + 1e-8)
  return adv, ret

def minibatch_indices(N, batch_size, shuffle=True):
  idx = np.arange(N)
  if shuffle:
    np.random.shuffle(idx)
  for start in range(0, N, batch_size):
    yield idx[start:start+batch_size]

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--sim_module", required=True, help="Path to the python file that defines TinyPhysicsModel/Simulator")
  parser.add_argument("--model_path", required=True, help="ONNX model path for TinyPhysicsModel")
  parser.add_argument("--data_path", required=True, help="CSV file or dir of CSVs")
  parser.add_argument("--save_path", default=str(Path(__file__).parent / "controllers" / "ppo_policy.pt"))
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--epochs", type=int, default=30)
  parser.add_argument("--gamma", type=float, default=0.99)
  parser.add_argument("--lam", type=float, default=0.95)
  parser.add_argument("--clip_eps", type=float, default=0.20)
  parser.add_argument("--pi_lr", type=float, default=3e-4)
  parser.add_argument("--vf_lr", type=float, default=1e-3)
  parser.add_argument("--entropy_coef", type=float, default=0.00)
  parser.add_argument("--batch_size", type=int, default=2048)
  parser.add_argument("--minibatch", type=int, default=256)
  parser.add_argument("--update_epochs", type=int, default=10)
  parser.add_argument("--exp", type=str, default='work_dirs')
  args = parser.parse_args()

  device = torch.device(args.device)
  sim_mod = load_sim(args.sim_module)
  writer = SummaryWriter(args.exp)

  # controller in training mode
  from controllers.ppo import Controller  # imports the file above
  ctrl = Controller(policy_path=args.save_path, training=True, device=args.device)

  # optimizer
  pi_params = list(ctrl.parameters())
  optimizer = optim.Adam([
      {"params": pi_params, "lr": args.pi_lr}
  ])

  # gather list of episodes (files)
  data_path = Path(args.data_path)
  if data_path.is_file():
    files = [data_path]
  else:
    files = sorted([p for p in data_path.iterdir() if p.suffix.lower()==".csv"])
  assert len(files) > 0, "No data files found."

  for epoch in range(1, args.epochs+1):
    all_obs, all_act, all_logp, all_rew, all_val, all_done = [], [], [], [], [], []
    # collect batch
    while sum(arr.shape[0] for arr in all_obs) < args.batch_size:
      f = files[np.random.randint(0, len(files))]
      obs, act, logp, rew, val, done = collect_episode(sim_mod, f, args.model_path, ctrl, device)
      # update obs normalizer
      ctrl.obs_rms.update(obs)
      all_obs.append(obs); all_act.append(act); all_logp.append(logp)
      all_rew.append(rew); all_val.append(val); all_done.append(done)

    obs = np.concatenate(all_obs, axis=0)
    act = np.concatenate(all_act, axis=0)
    logp_old = np.concatenate(all_logp, axis=0)
    rew = np.concatenate(all_rew, axis=0)
    val = np.concatenate(all_val, axis=0)
    done = np.concatenate(all_done, axis=0)

    adv, ret = compute_gae(rew, val, done, gamma=args.gamma, lam=args.lam)

    N = obs.shape[0]
    for _ in range(args.update_epochs):
      for idx in minibatch_indices(N, args.minibatch, shuffle=True):
        ob_mb = obs[idx]; ac_mb = act[idx]; lp_mb = logp_old[idx]
        adv_mb = adv[idx]; ret_mb = ret[idx]

        # recompute logp & value
        logp_new, v_new, entropy = ctrl.evaluate_actions(ob_mb, ac_mb)

        ratio = torch.exp(logp_new - torch.from_numpy(lp_mb).to(device))
        adv_t = torch.from_numpy(adv_mb).to(device)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0-args.clip_eps, 1.0+args.clip_eps) * adv_t
        pi_loss = -(torch.min(surr1, surr2)).mean()

        v_t = v_new
        ret_t = torch.from_numpy(ret_mb).to(device)
        vf_loss = ((v_t - ret_t).abs()).mean()

        ent = entropy.mean()
        pi_loss = 10 * pi_loss
        vf_loss = 0.001*vf_loss
        ent = args.entropy_coef*ent
        loss = pi_loss + vf_loss - ent
        writer.add_scalar("Loss/pi", (10 * pi_loss).item(), epoch)
        # print(pi_loss, 0.5*vf_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 0.5)
        optimizer.step()

    # save
    ctrl.save()

    with torch.no_grad():
      avg_ret = float(ret.mean())
      avg_adv = float(np.abs(adv).mean())
      print(f"[Epoch {epoch:03d}] steps={N}  return(mean)={avg_ret:.3f}  |adv|={avg_adv:.3f}  saved -> {args.save_path}")
      writer.add_scalar("Loss/pi", pi_loss.item(), epoch)
      writer.add_scalar("Loss/value", vf_loss.item(), epoch)
      writer.add_scalar("Return",avg_ret, epoch)
      writer.flush()
      

if __name__ == "__main__":
  main()