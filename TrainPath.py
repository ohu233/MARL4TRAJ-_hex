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
from utils.SoftActorCritic import ReplayBuffer, MLP, SACConfig, DiscreteSACAgent


@dataclass
class CurriculumConfig:
    num_stages: int = 4
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
    agent = DiscreteSACAgent(vec_dim=20, hex_radius=env.FOV, action_dim=action_dim,
                              cfg=cfg, use_gnn=use_gnn, in_channels=5)

    stage_trajs = env.split_traj_by_distance(curriculum_cfg.num_stages)
    stage_idx = 0
    env.set_curriculum_stage(stage_idx, stage_trajs[stage_idx], max_mode_count=4)
    env.set_mode_sampling_range(min_mode_count=1, max_mode_count=4)


    in_refine_phase = False
    stage_episode_count = 0
    stage_stable_count = 0
    refine_stable_count = 0

    total_steps = 0
    logs = []
    success_list = []
    match_list = [] # 存储匹配度
    trans_count_list = []
    traj_list = []  # 临时存储轨迹

    avg_reward_100_list = []
    reach_rate_100_list = []
    match_rate_100_list = []
    actor_loss_ep_list = []
    critic_loss_ep_list = []
    curriculum_stage_list = []
    mode_max_count_list = []
    refine_phase_list = []

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


    print(f"Curriculum enabled: {len(stage_trajs)} stages.")
    for sid, stage_df in enumerate(stage_trajs):
        from utils.hex_utils import hex_distance
        stage_dist = stage_df.apply(
            lambda r: hex_distance(
                (r['locxo'], r['locyo'], r['loczo']),
                (r['locxd'], r['locyd'], r['loczd']),
            ), axis=1
        )
        print(
            f"  stage={sid + 1}/{len(stage_trajs)}, samples={len(stage_df)}, "
            f"dist[min/mean/max]={stage_dist.min():.1f}/{stage_dist.mean():.1f}/{stage_dist.max():.1f}"
        )
    
    for ep in range(1, episodes + 1):
        s = env.reset()
        s_vec = state_to_vector(s)
        ep_reward = 0.0

        # 本回合 loss 累计
        ep_actor_losses = []
        ep_critic_losses = []

        for t in range(max_episode_steps):
            total_steps += 1

            if total_steps < cfg.start_steps:
                a = np.random.randint(action_dim)
            else:
                a = agent.select_action(s_vec, evaluate=False)

            ns, r, done, success = env.step(int(a))
            ns_vec = state_to_vector(ns)

            traj_list.append(env.hex_start)

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

        # 统计traj_list匹配度
        match_rate = calculate_match_rate(traj_list, env.multi_mapdata)
        traj_list.clear()
        logs.append(ep_reward)
        success_list.append(success)
        match_list.append(match_rate)
        trans_count_list.append(env.min_trans_count)

        window = curriculum_cfg.metrics_window
        avg_reward_100 = np.mean(logs[-window:])
        reach_rate_100 = sum(success_list[-window:]) / len(success_list[-window:]) * 100
        match_rate_100 = sum(match_list[-window:]) / len(match_list[-window:]) * 100

        avg_reward_100_list.append(avg_reward_100)
        reach_rate_100_list.append(reach_rate_100)
        match_rate_100_list.append(match_rate_100)
        curriculum_stage_list.append(stage_idx + 1)
        mode_max_count_list.append(env.max_mode_count)
        refine_phase_list.append(int(in_refine_phase))

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
                f"trans={env.min_trans_count}, "
                f"stage={stage_idx + 1}/{len(stage_trajs)}, "
                f"mode=random1-{env.max_mode_count}"
            )

        stage_episode_count += 1

        if not in_refine_phase:
            is_stage_stable = (
                reach_rate_100 >= curriculum_cfg.promote_reach_rate
                and match_rate_100 >= curriculum_cfg.promote_match_rate
            )
            is_refine_stable = (
                reach_rate_100 >= curriculum_cfg.refine_reach_rate
                and match_rate_100 >= curriculum_cfg.refine_match_rate
            )

            stage_stable_count = stage_stable_count + 1 if is_stage_stable else 0
            refine_stable_count = refine_stable_count + 1 if is_refine_stable else 0

            can_promote = (
                stage_episode_count >= curriculum_cfg.min_stage_episodes
                and stage_stable_count >= curriculum_cfg.promote_patience
            )

            if can_promote and stage_idx < len(stage_trajs) - 1:
                prev_stage = stage_idx
                stage_idx += 1
                curr_data = stage_trajs[stage_idx]
                prev_data = stage_trajs[prev_stage]
                n_mix = int(len(curr_data) * curriculum_cfg.prev_stage_mix_ratio)
                mixed = pd.concat(
                    [curr_data, prev_data.sample(n=min(n_mix, len(prev_data)), random_state=42)],
                    ignore_index=True,
                )
                env.set_curriculum_stage(stage_idx, mixed, max_mode_count=4)
                env.set_mode_sampling_range(min_mode_count=1, max_mode_count=4)
                stage_episode_count = 0
                stage_stable_count = 0
                refine_stable_count = 0
                print(
                    f"[Curriculum] Promote to distance stage {stage_idx + 1}/{len(stage_trajs)}"
                    f" (mixed {curriculum_cfg.prev_stage_mix_ratio*100:.0f}% from stage {prev_stage + 1}, random1-4)."
                )
            elif (
                can_promote
                and stage_idx == len(stage_trajs) - 1
                and stage_episode_count >= curriculum_cfg.min_refine_episodes
                and refine_stable_count >= curriculum_cfg.refine_patience
            ):
                in_refine_phase = True
                env.set_mode_sampling_range(min_mode_count=1, max_mode_count=3)
                stage_episode_count = 0
                stage_stable_count = 0
                refine_stable_count = 0
                print("[Curriculum] Final stage stable, switch map combo to random1-3.")

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
        "curriculum_stage": curriculum_stage_list,
        "mode_max_count": mode_max_count_list,
        "in_refine_phase": refine_phase_list,
        "min_trans_count": trans_count_list,
    })

    metrics_df.to_csv(f"PathModel/train_metrics{tag}.csv", index=False, encoding="utf-8")
    print(f"Saved metrics CSV to: PathModel/train_metrics{tag}.csv")

    return agent, logs


if __name__ == "__main__":

    from utils.hex_utils import load_hex_mapdata_raw

    mapdata = load_hex_mapdata_raw('data/hex_grid.pkl')
    traj = pd.read_csv('data//artificial_od_all.csv')

    # 对调换起点和终点进行训练，增加数据多样性
    # 六边形 cube 坐标: (locxo, locyo, loczo) ↔ (locxd, locyd, loczd)
    reversed_traj = traj.copy()
    for o_col, d_col in [('locxo', 'locxd'), ('locyo', 'locyd'), ('loczo', 'loczd')]:
        if o_col in reversed_traj.columns and d_col in reversed_traj.columns:
            reversed_traj[[o_col, d_col]] = reversed_traj[[d_col, o_col]].values
    traj = pd.concat([traj, reversed_traj], ignore_index=True)

    shuffled_traj = traj.sample(frac=1, random_state=42).reset_index(drop=True)

    # 课程学习开关
    train_mode = True
    curriculum_mode = True
    USE_GNN = True
    FOV = 3
    distance_threshold = 1.0
    env = PathEnv(train_mode=train_mode,
                  curriculum_mode=curriculum_mode,
                  mapdata=mapdata,
                  traj=shuffled_traj,
                  FOV=FOV,
                  distance_threshold=distance_threshold
                  )

    curriculum_cfg = CurriculumConfig(
        num_stages=4 if curriculum_mode else 1,  # 非课程学习时需要把该参数调整为1 否则为4
        metrics_window=100,
        min_stage_episodes=300,
        promote_reach_rate=85.0,
        promote_match_rate=0.0,
        promote_patience=3,
        min_refine_episodes=300,
        refine_reach_rate=85.0,
        refine_match_rate=0.0,
        refine_patience=4,
        prev_stage_mix_ratio=0.2,
    )

    agent, logs = train_sac_on_pathenv(env, episodes=5000, curriculum_cfg=curriculum_cfg, use_gnn=USE_GNN)