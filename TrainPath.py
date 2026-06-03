import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

from utils.Environment import PathEnv
from utils.tools import mapdata_to_modelmatrix, state_to_vector, calculate_match_rate
from utils.SoftActorCritic import ReplayBuffer, EpisodeBuffer, MLP, SACConfig, DiscreteSACAgent


@dataclass
class CurriculumConfig:
    distance_bins: int = None  # int → qcut 均分 N 段；list → cut 固定边界；None → 不分段
    metrics_window: int = 100
    min_stage_episodes: int = 300
    promote_reach_rate: float = 85
    promote_match_rate: float = 85
    promote_patience: int = 3
    min_refine_episodes: int = 600
    refine_reach_rate: float = 85.0
    refine_match_rate: float = 85.0
    refine_patience: int = 4
    prev_stage_mix_ratio: float = 0.2

def train_sac_on_pathenv(
    env,
    episodes: int = 500,
    max_episode_steps: int = 300,
    cfg: SACConfig = SACConfig(),
    curriculum_cfg: CurriculumConfig = CurriculumConfig(),
    seed: int = 42,
    use_gnn: bool = True,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    s0 = env.reset()
    s0_vec = state_to_vector(s0)
    state_dim = s0_vec.shape[0]
    env.traj_cnt = 0

    action_dim = 6
    agent = DiscreteSACAgent(vec_dim=23, hex_radius=env.FOV, action_dim=action_dim,
                              cfg=cfg, use_gnn=use_gnn, in_channels=5)

    env.set_mode_sampling_range(min_mode_count=1, max_mode_count=3)

    total_steps = 0
    logs = []
    success_list = []
    match_list = [] # 存储匹配度
    trans_count_list = []
    offroad_ratio_list = []
    offroad_streak_max_list = []
    no_progress_steps_list = []
    best_bfs_remaining_list = []
    done_reason_list = []
    traj_list = []  # 临时存储轨迹

    avg_reward_100_list = []
    reach_rate_100_list = []
    match_rate_100_list = []
    actor_loss_ep_list = []
    critic_loss_ep_list = []
    mode_max_count_list = []

    tag = ("_withGNN" if use_gnn else "") + ("_withCurri" if env.curriculum_mode else "")

    def _save_metrics_and_plots(ep):
        episodes_x = np.arange(1, len(logs) + 1)

        # Reward 曲线
        plt.figure(figsize=(8,4))
        plt.plot(episodes_x, logs, label="Episode Reward", alpha=0.5)
        plt.plot(episodes_x, avg_reward_100_list, label="Avg Reward (Last 100)", linewidth=2)
        plt.xlabel("Episode"); plt.ylabel("Reward")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"PathModel/reward_curves_ep{ep}{tag}.png", dpi=200)
        plt.close()

        # 到达率 + 匹配率
        plt.figure(figsize=(8,4))
        plt.plot(episodes_x, reach_rate_100_list, label="Reach Rate (Last 100) %", linewidth=2)
        plt.plot(episodes_x, match_rate_100_list, label="Match Rate (Last 100) %", linewidth=2)
        plt.xlabel("Episode"); plt.ylabel("Rate (%)")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"PathModel/rate_curves_ep{ep}{tag}.png", dpi=200)
        plt.close()

        # Actor/Critic Loss
        plt.figure(figsize=(8,4))
        plt.plot(episodes_x, actor_loss_ep_list, label="Actor Loss", linewidth=2)
        plt.plot(episodes_x, critic_loss_ep_list, label="Critic Loss", linewidth=2)
        plt.xlabel("Episode"); plt.ylabel("Loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"PathModel/loss_curves_ep{ep}{tag}.png", dpi=200)
        plt.close()


    for ep in range(1, episodes + 1):
        s = env.reset()
        s_vec = state_to_vector(s)
        ep_reward = 0.0
        traj_list.append(env.hex_start)

        # Episode-level HER support: collect transitions and push at episode end
        use_episode_buffer = isinstance(agent.replay, EpisodeBuffer)
        ep_transitions = [] if use_episode_buffer else None
        episode_meta = env.get_episode_metadata() if use_episode_buffer else None

        # 本回合 loss 累计
        ep_actor_losses = []
        ep_critic_losses = []

        for t in range(max_episode_steps):
            total_steps += 1

            in_random_exploration = total_steps < cfg.start_steps
            if hasattr(env, 'offroad_done_enabled'):
                env.offroad_done_enabled = not in_random_exploration

            if in_random_exploration:
                a = np.random.randint(action_dim)
            else:
                a = agent.select_action(s_vec, evaluate=False)

            # Capture the pre-action neighbor for HER reward recomputation
            pre_action_neighbor = env.neighbor.copy() if hasattr(env, 'neighbor') else None

            ns, r, done, success = env.step(int(a))
            ns_vec = state_to_vector(ns)

            traj_list.append(env.hex_start)

            if use_episode_buffer:
                ep_transitions.append((s_vec, int(a), float(r), ns_vec, bool(done), pre_action_neighbor))
            else:
                agent.replay.push(s_vec, a, r, ns_vec, float(done))
            s_vec = ns_vec
            ep_reward += float(r)

            if total_steps >= cfg.update_after and total_steps % cfg.update_every == 0:
                update_info = agent.update()
                if update_info:
                    ep_actor_losses.append(update_info["actor_loss"])
                    critic_loss = 0.5 * (update_info["q1_loss"] + update_info["q2_loss"])
                    ep_critic_losses.append(critic_loss)

            if done:
                break

        # Episode-level push for HER
        if use_episode_buffer and ep_transitions:
            agent.replay.push_episode(
                ep_transitions,
                hex_start=episode_meta['hex_start'],
                hex_end=episode_meta['hex_end'],
                bfs_dist=episode_meta['bfs_dist'],
            )

        # 统计traj_list匹配度
        match_rate = calculate_match_rate(traj_list, env.multi_mapdata)
        traj_list.clear()
        logs.append(ep_reward)
        success_list.append(success)
        match_list.append(match_rate)
        trans_count_list.append(env.min_trans_count)
        offroad_ratio_list.append(float(getattr(env, 'offroad_total', 0) / max(1, getattr(env, 'step_cnt', 1))))
        offroad_streak_max_list.append(int(getattr(env, 'offroad_streak_max', 0)))
        no_progress_steps_list.append(int(getattr(env, 'no_progress_steps', 0)))
        best_bfs_remaining_list.append(float(getattr(env, 'best_bfs_remaining', 0.0)))
        done_reason_list.append(getattr(env, 'done_reason', 'unknown'))

        window = curriculum_cfg.metrics_window
        avg_reward_100 = np.mean(logs[-window:])
        reach_rate_100 = sum(success_list[-window:]) / len(success_list[-window:]) * 100
        match_rate_100 = sum(match_list[-window:]) / len(match_list[-window:]) * 100

        avg_reward_100_list.append(avg_reward_100)
        reach_rate_100_list.append(reach_rate_100)
        match_rate_100_list.append(match_rate_100)
        mode_max_count_list.append(env.max_mode_count)

        actor_loss_ep = float(np.mean(ep_actor_losses)) if ep_actor_losses else np.nan
        critic_loss_ep = float(np.mean(ep_critic_losses)) if ep_critic_losses else np.nan

        actor_loss_ep_list.append(actor_loss_ep)
        critic_loss_ep_list.append(critic_loss_ep)

        if ep % 10 == 0:
            print(
                f"[Episode {ep:04d}] reward={ep_reward:.3f}, "
                f"average reward 100={avg_reward_100:.3f}, "
                f"reach rate={reach_rate_100:.2f}%, "
                f"match rate={match_rate_100:.2f}%, "
                f"trans={env.min_trans_count} "
            )

        if ep % 1000 == 0:
            torch.save(agent.actor.state_dict(), f"PathModel/sac_actor_ep{ep}{tag}.pth")
            _save_metrics_and_plots(ep)

    episodes_x = np.arange(1, len(logs) + 1)


    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, logs, label="Episode Reward", alpha=0.5)
    plt.plot(episodes_x, avg_reward_100_list, label="Avg Reward (Last 100)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"PathModel/reward_curves{tag}.png", dpi=200)
    plt.close()

    # 图2: 到达率 + 匹配率（均为100回合滑动窗口）
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, reach_rate_100_list, label="Reach Rate (Last 100) %", linewidth=2)
    plt.plot(episodes_x, match_rate_100_list, label="Match Rate (Last 100) %", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Rate (%)")
    plt.title("Training Reach/Match Rate Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"PathModel/rate_curves{tag}.png", dpi=200)
    plt.close()

    # 图3: Actor/Critic Loss
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, actor_loss_ep_list, label="Actor Loss", linewidth=2)
    plt.plot(episodes_x, critic_loss_ep_list, label="Critic Loss", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"PathModel/loss_curves{tag}.png", dpi=200)
    plt.close()

    metrics_df = pd.DataFrame({
        "episode": episodes_x,
        "episode_reward": logs,
        "avg_reward_100": avg_reward_100_list,
        "reach_rate_100": reach_rate_100_list,
        "match_rate_100": match_rate_100_list,
        "actor_loss": actor_loss_ep_list,
        "critic_loss": critic_loss_ep_list,
        "mode_max_count": mode_max_count_list,
        "min_trans_count": trans_count_list,
        "offroad_ratio": offroad_ratio_list,
        "offroad_streak_max": offroad_streak_max_list,
        "no_progress_steps": no_progress_steps_list,
        "best_bfs_remaining": best_bfs_remaining_list,
        "done_reason": done_reason_list,
    })

    metrics_df.to_csv(f"PathModel/train_metrics{tag}.csv", index=False, encoding="utf-8")
    print(f"Saved metrics CSV to: PathModel/train_metrics{tag}.csv")

    return agent, logs


if __name__ == "__main__":

    from utils.hex_utils import load_hex_mapdata_raw

    mapdata = load_hex_mapdata_raw('data/hex_grid.pkl')
    traj = pd.read_csv('data//artificial_od_single.csv')
    # 筛选掉起终点过近的样本（距离小于等于2个六边格），因为它们过于简单，无法提供有效的训练信号
    traj = traj[traj['distance_cells'] > 2]

    # 对调换起点和终点进行训练，增加数据多样性
    # 六边形 cube 坐标: (locxo, locyo, loczo) ↔ (locxd, locyd, loczd)
    reversed_traj = traj.copy()
    for o_col, d_col in [('locxo', 'locxd'), ('locyo', 'locyd'), ('loczo', 'loczd')]:
        if o_col in reversed_traj.columns and d_col in reversed_traj.columns:
            reversed_traj[[o_col, d_col]] = reversed_traj[[d_col, o_col]].values
    traj = pd.concat([traj, reversed_traj], ignore_index=True)

    shuffled_traj = traj.sample(frac=1, random_state=40).reset_index(drop=True)
    shuffled_traj = shuffled_traj[shuffled_traj['distance_cells'] > 4].reset_index(drop=True)
    
    # 课程学习开关
    train_mode = True
    curriculum_mode = False
    USE_GNN = False
    FOV = 5
    distance_threshold = 1.0
    env = PathEnv(train_mode=train_mode,
                  curriculum_mode=curriculum_mode,
                  mapdata=mapdata,
                  traj=shuffled_traj,
                  FOV=FOV,
                  distance_threshold=distance_threshold,
                  bfs_search_radius=30,
                  bfs_max_nodes=50000,
                  )

    curriculum_cfg = CurriculumConfig(
        distance_bins=None,
        metrics_window=100,
    )

    # HER 配置：默认 0.8 (Future 策略)，可改为 0.0 关闭
    sac_cfg = SACConfig(her_prob=0.3, her_strategy='future')

    agent, logs = train_sac_on_pathenv(env, episodes=5000, cfg=sac_cfg,
                                       curriculum_cfg=curriculum_cfg, use_gnn=USE_GNN)
