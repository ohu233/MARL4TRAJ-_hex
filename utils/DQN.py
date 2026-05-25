import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 200000

    start_steps: int = 2000
    train_freq: int = 1
    target_update_interval: int = 500

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50000

    hidden_dim: int = 256
    max_grad_norm: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _safe_get(d: Dict[str, Any], key: str, default=0.0):
    v = d.get(key, default)
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.array(v, dtype=np.float32).reshape(-1)
    return np.array([float(v)], dtype=np.float32)


def mode_state_to_vector(state: Dict[str, Any]) -> np.ndarray:
    """
    ModeEnv state 展平:
    previous: mode(4), match_rate(4), multi_match_rate, success, steps, path_len, time, distance, velocity
    current : mode(4), match_rate(4), multi_match_rate, success, steps, path_len, time, distance, velocity
    """
    prev = state.get("previous", {})
    cur = state.get("current", {})

    prev_vec = np.concatenate(
        [
            _safe_get(prev, "mode", [0, 0, 0, 0]),
            _safe_get(prev, "match_rate", [0, 0, 0, 0]),
            _safe_get(prev, "multi_match_rate", 0.0),
            _safe_get(prev, "success", 0.0),
            _safe_get(prev, "steps", 0.0),
            _safe_get(prev, "path_len", 0.0),
            _safe_get(prev, "time", 0.0),
            _safe_get(prev, "distance", 0.0),
            _safe_get(prev, "velocity", 0.0),
        ],
        axis=0,
    )

    cur_vec = np.concatenate(
        [
            _safe_get(cur, "mode", [0, 0, 0, 0]),
            _safe_get(cur, "match_rate", [0, 0, 0, 0]),
            _safe_get(cur, "multi_match_rate", 0.0),
            _safe_get(cur, "success", 0.0),
            _safe_get(cur, "steps", 0.0),
            _safe_get(cur, "path_len", 0.0),
            _safe_get(cur, "time", 0.0),
            _safe_get(cur, "distance", 0.0),
            _safe_get(cur, "velocity", 0.0),
        ],
        axis=0,
    )

    return np.concatenate([prev_vec, cur_vec], axis=0).astype(np.float32)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.action_dim = int(action_dim)

        self.q = QNet(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q_tgt = QNet(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval()

        self.optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.buffer_size)

        self.total_steps = 0

    def epsilon(self) -> float:
        t = min(self.total_steps, self.cfg.eps_decay_steps)
        frac = t / float(max(self.cfg.eps_decay_steps, 1))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def select_action(self, state_vec: np.ndarray, evaluate: bool = False) -> int:
        if (not evaluate) and (random.random() < self.epsilon()):
            return random.randint(0, self.action_dim - 1)

        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(s)
        a = int(torch.argmax(q, dim=-1).item())
        return a

    def update(self):
        if len(self.replay) < self.cfg.batch_size:
            return {}

        s, a, r, ns, d = self.replay.sample(self.cfg.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(-1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(-1)

        q_sa = self.q(s).gather(1, a)

        with torch.no_grad():
            # Double DQN: next action from online net, next value from target net
            next_a = torch.argmax(self.q(ns), dim=-1, keepdim=True)
            next_q = self.q_tgt(ns).gather(1, next_a)
            y = r + (1.0 - d) * self.cfg.gamma * next_q

        loss = F.smooth_l1_loss(q_sa, y)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.optim.step()

        return {"loss": float(loss.item())}

    def maybe_update_target(self):
        if self.total_steps % self.cfg.target_update_interval == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

    def save(self, path: str):
        torch.save(self.q.state_dict(), path)

    def load(self, path: str, map_location=None):
        sd = torch.load(path, map_location=map_location if map_location is not None else self.device)
        self.q.load_state_dict(sd)
        self.q_tgt.load_state_dict(sd)


def train_dqn_on_modeenv(
    env,
    episodes: int = 5000,
    max_episode_steps: int = 20,
    cfg: DQNConfig = DQNConfig(),
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    s0 = env.reset()
    s0_vec = mode_state_to_vector(s0)

    state_dim = s0_vec.shape[0]
    action_dim = 15  # 4bit mode 组合 (1~15, 排除全零)

    agent = DQNAgent(state_dim, action_dim, cfg)

    logs = []
    success_logs = []
    loss_logs = []

    for ep in range(1, episodes + 1):
        s = env.reset()
        s_vec = mode_state_to_vector(s)

        ep_reward = 0.0
        ep_success = 0
        ep_losses = []

        for _ in range(max_episode_steps):
            agent.total_steps += 1

            if agent.total_steps < cfg.start_steps:
                a = random.randint(0, action_dim - 1)
            else:
                a = agent.select_action(s_vec, evaluate=False)

            ns, r, done, succ = env.step(a)
            ns_vec = mode_state_to_vector(ns)

            agent.replay.push(s_vec, a, float(r), ns_vec, float(done))
            s_vec = ns_vec

            if agent.total_steps % cfg.train_freq == 0:
                info = agent.update()
                if info:
                    ep_losses.append(info["loss"])

            agent.maybe_update_target()

            ep_reward += float(r)
            ep_success = max(ep_success, int(succ))

            if done:
                break

        logs.append(ep_reward)
        success_logs.append(ep_success)
        loss_logs.append(float(np.mean(ep_losses)) if ep_losses else np.nan)

        if ep % 20 == 0:
            avg_r = float(np.mean(logs[-100:]))
            succ_r = float(np.mean(success_logs[-100:]) * 100.0)
            print(
                f"[Ep {ep:05d}] "
                f"ep_reward={ep_reward:.3f}, avg100={avg_r:.3f}, "
                f"succ100={succ_r:.2f}%, eps={agent.epsilon():.3f}"
            )

    history = {
        "episode_reward": logs,
        "episode_success": success_logs,
        "episode_loss": loss_logs,
    }
    return agent, history