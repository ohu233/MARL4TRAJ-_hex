import random
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hex_utils import (
    get_fixed_edge_index, hex_distance, ACTION_TO_HEX_IDX, _hex_ring_offsets,
)


# ============================================================
# GNN 六边形 Patch 编码器
# ============================================================
class HexGraphConv(nn.Module):
    """单层图卷积（GCN），不依赖 torch_geometric。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, a_hat):
        """
        x:      (B, N, in_dim)  节点特征
        a_hat:  (N, N)          归一化邻接矩阵（含自环）
        返回:   (B, N, out_dim)
        """
        # 邻居聚合: A_hat @ x
        support = torch.bmm(a_hat.unsqueeze(0).expand(x.size(0), -1, -1), x)
        return support @ self.weight + self.bias


class HexPatchEncoder(nn.Module):
    """将六边形 patch 编码为固定维度向量。"""

    def __init__(self, in_channels: int = 6, hidden_dim: int = 32,
                 out_dim: int = 64, num_layers: int = 2,
                 n_nodes: int = 37):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_nodes = n_nodes
        self.radius = int((np.sqrt(12 * n_nodes - 3) - 3) / 6)
        self.num_vectors = self.radius + 7 if self.radius >= 1 else 2

        layers = []
        cur_dim = in_channels
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else out_dim
            layers.append(HexGraphConv(cur_dim, next_dim))
            cur_dim = next_dim
        self.convs = nn.ModuleList(layers)

        # 注册固定邻接矩阵（不参与训练）
        edge_index = get_fixed_edge_index(self.radius)
        a = torch.zeros(n_nodes, n_nodes)
        a[edge_index[0], edge_index[1]] = 1.0
        a = a + torch.eye(n_nodes)  # 加自环
        deg = a.sum(dim=1).pow(-0.5)
        deg[torch.isinf(deg)] = 0
        a_hat = deg.unsqueeze(1) * a * deg.unsqueeze(0)
        self.register_buffer('a_hat', a_hat)

    def forward(self, x):
        """
        x: (B, N, in_channels) 节点特征
        返回: (B, num_vectors * out_dim)
          ring0 + ring1(6cells) + ring2..R(各1 mean) + global
        """
        for conv in self.convs:
            x = F.relu(conv(x, self.a_hat))

        pooled = [x[:, 0:1, :]]   # ring0: center
        pooled.append(x[:, 1:7, :])  # ring1: 保留 6 个 cell（方向信息）

        idx = 7
        for r in range(2, self.radius + 1):
            n = 6 * r
            pooled.append(x[:, idx:idx + n, :].mean(dim=1, keepdim=True))
            idx += n

        pooled.append(x.mean(dim=1, keepdim=True))  # global pool

        combined = torch.cat(pooled, dim=1)
        return combined.reshape(x.size(0), -1)


# ============================================================
# Replay Buffer (with HER support)
# ============================================================
class ReplayBuffer:
    """Original transition-level replay buffer (kept for backward compat)."""

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


class EpisodeBuffer:
    """Episode-level replay buffer with optional Hindsight Experience Replay (HER).

    Each pushed episode stores:
      - transitions: list of (s, a, r, ns, done, old_neighbor)
      - hex_start:  absolute cube coords of trajectory origin
      - hex_end:    absolute cube coords of trajectory destination (original goal)
      - bfs_dist:   BFS distance field built from hex_end (or None)
      - achieved_goals: list of absolute cube positions visited (for HER sampling)
    """

    def __init__(self, capacity_episodes: int, her_prob: float = 0.8,
                 her_strategy: str = 'future'):
        self.episodes = deque(maxlen=capacity_episodes)
        self.her_prob = her_prob
        self.her_strategy = her_strategy

    def push_episode(self, transitions: list, hex_start: tuple,
                     hex_end: tuple, bfs_dist: dict = None):
        """Push a completed episode. transitions is a list of
        (s, a, r, ns, done, old_neighbor) tuples (no requirement on field order
        beyond what _relabel_transition expects).

        NOTE: bfs_dist is accepted for API compatibility but NOT stored, because
        each BFS field is up to 50K cells (>> memory) and HER falls back to
        hex_distance anyway. Keeping hex_start/hex_end/achieved_goals is enough.
        """
        del bfs_dist  # accepted for API compat; not stored to save memory
        # Compute achieved_goals: absolute positions visited
        achieved_goals = []
        for t in transitions:
            s = t[0]
            remaining = s[0:3]
            achieved_goals.append(_remaining_to_abs(remaining, hex_end))
        # Also include the final next_state's absolute position
        if transitions:
            last_t = transitions[-1]
            ns = last_t[3]
            remaining = ns[0:3]
            achieved_goals.append(_remaining_to_abs(remaining, hex_end))

        self.episodes.append({
            'transitions': transitions,
            'hex_start': tuple(hex_start),
            'hex_end': tuple(hex_end),
            'bfs_dist': None,  # not stored to save memory
            'achieved_goals': achieved_goals,
        })

    def __len__(self):
        return sum(len(ep['transitions']) for ep in self.episodes)

    def sample(self, batch_size: int):
        """Sample batch_size transitions, optionally with HER relabeling.

        Returns: (s, a, r, ns, d) numpy arrays, or None if not enough data.
        """
        all_transitions = []
        for ep_idx, ep in enumerate(self.episodes):
            for t_idx, t in enumerate(ep['transitions']):
                all_transitions.append((ep_idx, t_idx, t, ep))

        if len(all_transitions) < batch_size:
            return None

        sampled = random.sample(all_transitions, batch_size)

        new_s, new_a, new_r, new_ns, new_d = [], [], [], [], []

        for ep_idx, t_idx, t, ep in sampled:
            if random.random() < self.her_prob:
                relabeled = self._relabel_transition(t, t_idx, ep)
                if relabeled is None:
                    # No future goal available: keep original
                    new_s.append(t[0]); new_a.append(t[1])
                    new_r.append(t[2]); new_ns.append(t[3])
                    new_d.append(float(t[4]))
                else:
                    ns, a, r, nns, d = relabeled
                    new_s.append(ns); new_a.append(a)
                    new_r.append(r); new_ns.append(nns)
                    new_d.append(float(d))
            else:
                new_s.append(t[0]); new_a.append(t[1])
                new_r.append(t[2]); new_ns.append(t[3])
                new_d.append(float(t[4]))

        return (
            np.array(new_s, dtype=np.float32),
            np.array(new_a, dtype=np.int64),
            np.array(new_r, dtype=np.float32),
            np.array(new_ns, dtype=np.float32),
            np.array(new_d, dtype=np.float32),
        )

    def _relabel_transition(self, t, t_idx, ep):
        """Relabel a transition with a future achieved goal. Returns
        (new_s, a, new_r, new_ns, new_done) or None if no future goal."""
        achieved = ep['achieved_goals']
        # Future strategy: pick a goal from states STRICTLY AFTER t_idx
        future_goals = achieved[t_idx + 1:]
        if not future_goals:
            return None

        new_goal = tuple(random.choice(future_goals))
        s, a, _, ns, done, old_neighbor = t  # original reward (idx 2) is recomputed below
        hex_start = ep['hex_start']
        bfs_dist = ep['bfs_dist']

        # Recover absolute positions from remaining-distance vectors.
        cur_abs = _remaining_to_abs(s[0:3], ep['hex_end'])
        prev_abs = _remaining_to_abs(s[3:6], ep['hex_end'])
        next_abs = _remaining_to_abs(ns[0:3], ep['hex_end'])

        # New remaining-distance fields to the fake goal
        new_remaining = _hex_sub(new_goal, cur_abs)
        new_prev_remaining = _hex_sub(new_goal, prev_abs)  # offset from previous pos to new goal
        new_next_remaining = _hex_sub(new_goal, next_abs)
        new_next_prev_remaining = _hex_sub(new_goal, cur_abs)  # for next state, prev = current

        # New normalized BFS distance (fallback to hex distance because BFS fields are not stored).
        new_cur_bfs = _lookup_bfs(cur_abs, new_goal, bfs_dist, hex_end=ep['hex_end'])
        new_total_bfs = _lookup_bfs(hex_start, new_goal, bfs_dist, hex_end=ep['hex_end'])
        new_next_bfs = _lookup_bfs(next_abs, new_goal, bfs_dist, hex_end=ep['hex_end'])
        new_cur_normalized = new_cur_bfs / max(1.0, new_total_bfs)
        new_next_normalized = new_next_bfs / max(1.0, new_total_bfs)

        # Rebuild state vector (matches state_to_vector layout)
        # [0:3]   remaining_distance
        # [3:6]   previous_remaining_distance
        # [6]     normalized_bfs_remaining
        # [7:11]  mode onehot
        # [11:]   patch (last channel is goal-dependent BFS gradient)
        new_s_patch = _replace_goal_gradient_channel(s[11:], cur_abs, new_goal)
        new_ns_patch = _replace_goal_gradient_channel(ns[11:], next_abs, new_goal)
        new_s = np.concatenate([
            np.asarray(new_remaining, dtype=np.float32),
            np.asarray(new_prev_remaining, dtype=np.float32),
            np.asarray([new_cur_normalized], dtype=np.float32),
            np.asarray(s[7:11], dtype=np.float32),
            new_s_patch,
        ]).astype(np.float32)

        new_ns = np.concatenate([
            np.asarray(new_next_remaining, dtype=np.float32),
            np.asarray(new_next_prev_remaining, dtype=np.float32),
            np.asarray([new_next_normalized], dtype=np.float32),
            np.asarray(ns[7:11], dtype=np.float32),
            new_ns_patch,
        ]).astype(np.float32)

        # Recompute reward
        new_r = _compute_her_reward(new_cur_bfs, new_next_bfs, old_neighbor, int(a))

        # Determine done: if next_abs is within 1 hex step of new_goal
        if hex_distance(next_abs, new_goal) <= 1:
            new_done = True
            new_r += 50.0  # success bonus
        else:
            new_done = bool(done)

        return (new_s, int(a), float(new_r), new_ns, new_done)


# ============================================================
# HER helper functions
# ============================================================
def _remaining_to_abs(remaining, hex_end):
    """Convert a goal-minus-position vector to absolute cube coords."""
    return (
        hex_end[0] - remaining[0],
        hex_end[1] - remaining[1],
        hex_end[2] - remaining[2],
    )


def _hex_sub(c1, c2):
    """Cube subtraction (imported lazily to avoid circular imports)."""
    return (c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2])


def _replace_goal_gradient_channel(patch, pos, goal):
    """Rebuild the final patch channel for a HER-relabeled goal.

    HER does not store a new road BFS field, so this uses cube-distance descent
    on cells marked as road by the first patch channel.
    """
    patch = np.asarray(patch, dtype=np.float32).copy()
    if patch.size == 0 or patch.size % 6 != 0:
        return patch

    n_cells = patch.size // 6
    road_patch = patch[:n_cells]
    offsets = []
    radius = 0
    while len(offsets) < n_cells:
        offsets.extend(_hex_ring_offsets(radius))
        radius += 1

    current_dist = float(hex_distance(pos, goal))
    gradient = np.full(n_cells, -1.0, dtype=np.float32)
    for idx, (dq, dr, ds) in enumerate(offsets[:n_cells]):
        if road_patch[idx] == 0:
            continue
        cell = (pos[0] + dq, pos[1] + dr, pos[2] + ds)
        gradient[idx] = float(np.clip(
            current_dist - hex_distance(cell, goal),
            -1.0,
            1.0,
        ))

    patch[5 * n_cells:6 * n_cells] = gradient
    return patch


def _lookup_bfs(pos, goal, bfs_dist, hex_end):
    """Look up BFS distance from pos to goal.

    bfs_dist: dict keyed by road cell, value = BFS distance from that cell to hex_end.
    Returns a lower-bound estimate: hex_distance if either pos or goal is off-road.
    """
    if bfs_dist is None:
        return float(hex_distance(pos, goal))
    # For speed, use hex_distance as the distance estimate
    # (full BFS path: pos → road_start → road_end → goal, requires more work)
    return float(hex_distance(pos, goal))


def _compute_her_reward(prev_dist, curr_dist, neighbor, action):
    """Recompute reward for a relabeled transition. Mirrors PathEnv.calculate_reward."""
    is_on_road = False
    if neighbor is not None:
        is_on_road = neighbor[ACTION_TO_HEX_IDX[action]] != 0
    dist_change = prev_dist - curr_dist

    if dist_change > 0:
        reward = 1.0 * dist_change
        reward += 0.3 if is_on_road else -1.3
    else:
        reward = -1.0
        reward += 0.5 if is_on_road else -1.5
    return float(reward)


# ============================================================
# MLP
# ============================================================
class MLP(nn.Module):
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


# ============================================================
# State Encoder（hex + GNN 版本）
# ============================================================
class StateEncoder(nn.Module):
    """将 flat state 拆分为 vector 特征 + hex patch，用 GNN 编码 patch。"""

    def __init__(self, vec_dim: int = 11, hex_radius: int = 3,
                 use_gnn: bool = True, in_channels: int = 6):
        super().__init__()
        self.vec_dim = vec_dim
        self.hex_radius = hex_radius
        self.use_gnn = use_gnn
        self.in_channels = in_channels
        self.n_cells = 3 * hex_radius**2 + 3 * hex_radius + 1

        if use_gnn:
            self.patch_encoder = HexPatchEncoder(
                in_channels=in_channels, hidden_dim=32, out_dim=64,
                n_nodes=self.n_cells,
            )
            self.out_dim = vec_dim + self.patch_encoder.num_vectors * 64
        else:
            self.patch_encoder = None
            self.out_dim = vec_dim + in_channels * self.n_cells

    def forward(self, x):
        vec = x[:, :self.vec_dim]
        patch = x[:, self.vec_dim:]

        if self.use_gnn and self.patch_encoder is not None:
            patch = patch.view(-1, self.in_channels, self.n_cells).transpose(1, 2).contiguous()
            patch_feat = self.patch_encoder(patch)
        else:
            patch_feat = patch

        return torch.cat([vec, patch_feat], dim=1)


# ============================================================
# PolicyNet
# ============================================================
class PolicyNet(nn.Module):
    """StateEncoder + MLP head，用于 actor / Q 网络。"""

    def __init__(self, vec_dim: int, hex_radius: int, out_dim: int,
                 hidden_dim: int = 256, use_gnn: bool = True,
                 in_channels: int = 6):
        super().__init__()
        self.encoder = StateEncoder(vec_dim, hex_radius, use_gnn, in_channels)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


# ============================================================
# SAC Config
# ============================================================
@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 128
    buffer_size: int = 200000
    start_steps: int = 2000
    update_after: int = 1000
    update_every: int = 1
    hidden_dim: int = 256
    target_entropy_ratio: float = 0.85
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # HER (Hindsight Experience Replay) settings
    her_prob: float = 0.0  # 0.0 disables HER; 0.8 recommended for sparse-reward long-distance
    her_strategy: str = 'future'  # 'future' | 'final' | 'episode'
    buffer_episodes: int = 500  # EpisodeBuffer capacity (episodes, not transitions)


# ============================================================
# Discrete SAC Agent
# ============================================================
class DiscreteSACAgent:
    def __init__(self, vec_dim: int = 11, hex_radius: int = 3,
                 action_dim: int = 6,
                 cfg: SACConfig = None, use_gnn: bool = True,
                 in_channels: int = 6):
        if cfg is None:
            cfg = SACConfig()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.action_dim = action_dim
        self.hex_radius = hex_radius
        self.use_gnn = use_gnn

        def _make_net(out_dim):
            return PolicyNet(vec_dim, hex_radius, out_dim,
                             cfg.hidden_dim, use_gnn, in_channels).to(self.device)

        self.actor = _make_net(action_dim)
        self.q1 = _make_net(action_dim)
        self.q2 = _make_net(action_dim)
        self.q1_target = _make_net(action_dim)
        self.q2_target = _make_net(action_dim)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=cfg.lr)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=cfg.lr)

        # 自动温度
        self.log_alpha = torch.tensor(np.log(0.1), dtype=torch.float32,
                                       requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        # 离散动作目标熵
        self.target_entropy = cfg.target_entropy_ratio * np.log(action_dim)

        # Use EpisodeBuffer when HER is enabled, else standard ReplayBuffer
        if cfg.her_prob > 0.0:
            self.replay = EpisodeBuffer(
                capacity_episodes=cfg.buffer_episodes,
                her_prob=cfg.her_prob,
                her_strategy=cfg.her_strategy,
            )
        else:
            self.replay = ReplayBuffer(cfg.buffer_size)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state_vec: np.ndarray, evaluate: bool = False) -> int:
        s = torch.tensor(state_vec, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        logits = self.actor(s)
        probs = F.softmax(logits, dim=-1)

        if evaluate:
            a = torch.argmax(probs, dim=-1).item()
        else:
            dist = torch.distributions.Categorical(probs=probs)
            a = dist.sample().item()
        return int(a)

    def update(self):
        if len(self.replay) < self.cfg.batch_size:
            return {}

        s, a, r, ns, d = self.replay.sample(self.cfg.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(-1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # ===== Q target =====
        with torch.no_grad():
            next_logits = self.actor(ns)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_probs = next_log_probs.exp()

            q1_t = self.q1_target(ns)
            q2_t = self.q2_target(ns)
            q_t_min = torch.min(q1_t, q2_t)

            next_v = (next_probs * (q_t_min - self.alpha.detach() * next_log_probs)
                      ).sum(dim=-1, keepdim=True)
            q_target = r + (1.0 - d) * self.cfg.gamma * next_v

        q1_all = self.q1(s)
        q2_all = self.q2(s)
        q1_sa = q1_all.gather(1, a)
        q2_sa = q2_all.gather(1, a)

        q1_loss = F.mse_loss(q1_sa, q_target)
        q2_loss = F.mse_loss(q2_sa, q_target)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 10.0)
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 10.0)
        self.q2_optim.step()

        # ===== Actor =====
        logits = self.actor(s)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        with torch.no_grad():
            q_min = torch.min(self.q1(s), self.q2(s))

        actor_loss = (probs * (self.alpha.detach() * log_probs - q_min)
                      ).sum(dim=-1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optim.step()

        # ===== Alpha =====
        entropy = -(probs * log_probs).sum(dim=-1)
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ===== soft update =====
        self._soft_update(self.q1, self.q1_target, self.cfg.tau)
        self._soft_update(self.q2, self.q2_target, self.cfg.tau)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    @staticmethod
    def _soft_update(net, target_net, tau):
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
