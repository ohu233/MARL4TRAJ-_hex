import random
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hex_utils import get_fixed_edge_index


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

    def __init__(self, in_channels: int = 5, hidden_dim: int = 32,
                 out_dim: int = 64, num_layers: int = 2,
                 n_nodes: int = 37):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_nodes = n_nodes

        layers = []
        cur_dim = in_channels
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else out_dim
            layers.append(HexGraphConv(cur_dim, next_dim))
            cur_dim = next_dim
        self.convs = nn.ModuleList(layers)

        # 注册固定邻接矩阵（不参与训练）
        edge_index = get_fixed_edge_index(int((np.sqrt(12 * n_nodes - 3) - 3) / 6))
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
        返回: (B, 8 * out_dim) — 中心 + 6方向 + 全局池化
        """
        for conv in self.convs:
            x = F.relu(conv(x, self.a_hat))

        center = x[:, 0:1, :]                     # (B, 1, out_dim) 中心节点
        ring1 = x[:, 1:7, :]                      # (B, 6, out_dim) ring-1
        global_pool = x.mean(dim=1, keepdim=True) # (B, 1, out_dim) 全局平均

        combined = torch.cat([center, ring1, global_pool], dim=1)  # (B, 8, out_dim)
        return combined.reshape(x.size(0), -1)    # (B, 8 * out_dim)


# ============================================================
# Replay Buffer
# ============================================================
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

    def __init__(self, vec_dim: int = 20, hex_radius: int = 3,
                 use_gnn: bool = True, in_channels: int = 5):
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
            self.out_dim = vec_dim + 8 * 64  # center + ring1(6) + global_pool
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
                 in_channels: int = 5):
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


# ============================================================
# Discrete SAC Agent
# ============================================================
class DiscreteSACAgent:
    def __init__(self, vec_dim: int = 20, hex_radius: int = 3,
                 action_dim: int = 6,
                 cfg: SACConfig = None, use_gnn: bool = True,
                 in_channels: int = 5):
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
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
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
