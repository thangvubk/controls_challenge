import math
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

try:
  from controllers import BaseController  # your existing interface
except Exception:
  class BaseController:  # fallback for type checking
    def update(self, target, current, state, future_plan=None):
      raise NotImplementedError

STEER_LOW, STEER_HIGH = -2.0, 2.0
STEER_HALF_RANGE = (STEER_HIGH - STEER_LOW) / 2.0
STEER_CENTER = (STEER_HIGH + STEER_LOW) / 2.0

# ---------- small utils ----------
class RunningNorm:
  def __init__(self, eps: float = 1e-6):
    self.mean = None
    self.var = None
    self.count = eps

  def update(self, x: np.ndarray):
    x = x.astype(np.float64)
    if self.mean is None:
      self.mean = x.mean(axis=0)
      self.var = x.var(axis=0) + 1e-6
      self.count = x.shape[0]
    else:
      batch_mean = x.mean(axis=0)
      batch_var = x.var(axis=0) + 1e-6
      batch_count = x.shape[0]
      delta = batch_mean - self.mean
      tot = self.count + batch_count
      new_mean = self.mean + delta * (batch_count / tot)
      m_a = self.var * self.count
      m_b = batch_var * batch_count
      M2 = m_a + m_b + (delta**2) * (self.count * batch_count / tot)
      new_var = M2 / tot
      self.mean, self.var, self.count = new_mean, new_var, tot

  def normalize(self, x: np.ndarray):
    if self.mean is None:
      return x
    return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

def _tanh_squash(mu: torch.Tensor, log_std: torch.Tensor, x: torch.Tensor = None):
  std = torch.exp(log_std)
  dist = D.Normal(mu, std)
  if x is None:
    z = dist.rsample()
  else:
    # inverse tanh for deterministic eval when an action is given in pre-squash space (not needed)
    z = x
  a = torch.tanh(z)
  # log_prob with squash correction
  logp = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
  return a, logp.sum(-1, keepdim=True)

# ---------- model ----------
class ActorCritic(nn.Module):
  def __init__(self, obs_dim: int, hidden: Tuple[int,int]=(32,32)):
    super().__init__()
    self.pi = nn.Sequential(
      nn.Linear(obs_dim, hidden[0]), nn.ReLU(),
      nn.Linear(hidden[1], 1)  # mean (pre-squash)
    )
    self.log_std = nn.Parameter(torch.zeros(1))  # shared log_std (simple & stable)
    self.v  = nn.Sequential(
      nn.Linear(obs_dim, hidden[0]), nn.ReLU(),
      nn.Linear(hidden[1], 1)
    )

  def act(self, obs: torch.Tensor, deterministic: bool):
    mu = self.pi(obs)
    if deterministic:
      a = torch.tanh(mu)
      logp = None
    else:
      a, logp = _tanh_squash(mu, self.log_std)
    return a, logp

  def value(self, obs: torch.Tensor):
    return self.v(obs)

# ---------- controller ----------
class Controller(BaseController):
  """
  PPO policy head that plugs into TinyPhysicsSimulator via BaseController.
  - During eval (default), it loads weights and acts deterministically.
  - During training, it samples actions and stores (obs, act, logp) for the trainer.
  """
  def __init__(self, policy_path: Optional[str] = None, training: bool = False, device: str = "cpu"):
    self.training_mode = training
    self.device = torch.device(device)
    self.prev_action = 0.0

    # feature normalizer (updated by trainer; here we keep one for eval safety)
    self.obs_rms = RunningNorm()

    # feature dimension (see _build_obs below)
    self.obs_dim = 11

    self.ac = ActorCritic(self.obs_dim).to(self.device)
    self.ckpt_path = Path(policy_path) if policy_path else Path(__file__).with_name("ppo_policy.pt")
    self._maybe_load()

    # buffers for training
    self._obs_buf, self._act_buf, self._logp_buf = [], [], []

  def _maybe_load(self):
    if self.ckpt_path.exists():
      data = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
      self.ac.load_state_dict(data["model"])
      # restore normalizer if present
      if "obs_mean" in data and "obs_var" in data and "obs_count" in data:
        self.obs_rms.mean  = data["obs_mean"]
        self.obs_rms.var   = data["obs_var"]
        self.obs_rms.count = data["obs_count"]
    else:
      # fresh weights (okay for training init; for eval it will act but untrained)
      pass

  def save(self):
    self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
      "model": self.ac.state_dict(),
      "obs_mean": self.obs_rms.mean,
      "obs_var": self.obs_rms.var,
      "obs_count": self.obs_rms.count,
    }, self.ckpt_path)

  def reset(self):
    self.prev_action = 0.0
    self._obs_buf.clear()
    self._act_buf.clear()
    self._logp_buf.clear()

  # --- featurization: compact, informative, stable ---
  @staticmethod
  def _stats(x: list, n: int = 10):
    arr = np.asarray(x[:n], dtype=np.float32)
    if arr.size == 0:
      return 0.0, 0.0
    return float(arr.mean()), float(arr.std() + 1e-6)

  def _build_obs(self, target: float, current: float, state, future_plan) -> np.ndarray:
    # future summaries (first ~1s @ 10Hz)
    m_tlat, s_tlat = self._stats(getattr(future_plan, "lataccel", []), 10)
    m_rroll, s_rroll = self._stats(getattr(future_plan, "roll_lataccel", []), 10)

    # features
    obs = np.array([
      float(target),
      float(current),
      float(target - current),
      float(state.roll_lataccel),
      float(state.v_ego),
      float(state.a_ego),
      float(self.prev_action),
      m_tlat, s_tlat,
      m_rroll, s_rroll
    ], dtype=np.float32)
    # normalize (no-op if stats not set)
    obs = self.obs_rms.normalize(obs)
    return obs

  # called by TinyPhysicsSimulator
  def update(self, target, current, state, future_plan=None) -> float:
    obs_np = self._build_obs(target, current, state, future_plan)
    obs = torch.from_numpy(obs_np).to(self.device).unsqueeze(0).float()
    deterministic = not self.training_mode
    a_tanh, logp = self.ac.act(obs, deterministic=deterministic)  # in [-1,1]
    # scale to steer range
    act = (a_tanh.detach().cpu().numpy()[0,0] * STEER_HALF_RANGE) + STEER_CENTER
    act = float(np.clip(act, STEER_LOW, STEER_HIGH))

    # buffers for trainer
    if self.training_mode:
      self._obs_buf.append(obs_np)
      self._act_buf.append([act])
      self._logp_buf.append(float(logp.detach().cpu().numpy()[0,0]))

    self.prev_action = act
    return act

  # --- interfaces trainer will use ---
  def pop_trajectory(self):
    """
    Returns and clears buffered (obs, act, logp) for the last episode.
    """
    obs = np.asarray(self._obs_buf, dtype=np.float32)
    act = np.asarray(self._act_buf, dtype=np.float32)
    logp = np.asarray(self._logp_buf, dtype=np.float32)
    self._obs_buf.clear(); self._act_buf.clear(); self._logp_buf.clear()
    return obs, act, logp

  @torch.no_grad()
  def value_batch(self, obs_np: np.ndarray) -> np.ndarray:
    obs = torch.from_numpy(obs_np).to(self.device)
    v = self.ac.value(obs).cpu().numpy()
    return v

  def evaluate_actions(self, obs_np: np.ndarray, act_np: np.ndarray):
    """
    For PPO update: recompute logp and value under current params.
    """
    obs = torch.from_numpy(obs_np).to(self.device)
    act = torch.from_numpy(act_np).to(self.device)
    # map action back to tanh-space target via inverse scaling, then atanh
    a_unit = torch.clamp((act - STEER_CENTER) / STEER_HALF_RANGE, -0.999, 0.999)
    z = torch.atanh(a_unit)  # pre-squash

    mu = self.ac.pi(obs)
    std = torch.exp(self.ac.log_std)
    dist = D.Normal(mu, std)
    # squash correction
    logp = dist.log_prob(z) - torch.log(1 - torch.tanh(z).pow(2) + 1e-6)
    logp = logp.sum(-1, keepdim=True)
    v = self.ac.value(obs)
    entropy = dist.entropy().sum(-1, keepdim=True)
    return logp, v, entropy

  def parameters(self):
    return self.ac.parameters()