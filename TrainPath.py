import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from utils.Environment import PathEnv
from utils.tools import state_to_vector
from utils.SoftActorCritic import EpisodeBuffer, SACConfig, DiscreteSACAgent


def train_sac_on_pathenv(
    env,
    episodes: int = 500,
    max_episode_steps: int = 300,
    cfg: SACConfig = SACConfig(),
    metrics_window: int = 100,
    seed: int = 42,
    use_gnn: bool = True,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env.reset()
    env.traj_cnt = 0

    action_dim = 6
    agent = DiscreteSACAgent(vec_dim=11, hex_radius=env.FOV, action_dim=action_dim,
                              cfg=cfg, use_gnn=use_gnn, in_channels=6)

    env.set_mode_sampling_range(min_mode_count=1, max_mode_count=3)

    total_steps = 0
    logs = []
    success_list = []
    match_list = [] # 存储匹配度

    avg_reward_window_list = []
    reach_rate_window_list = []
    match_rate_window_list = []
    actor_loss_ep_list = []
    critic_loss_ep_list = []

    tag = ("_withGNN" if use_gnn else "")

    def _save_metrics_and_plots(ep):
        episodes_x = np.arange(1, len(logs) + 1)

        # Reward 曲线
        plt.figure(figsize=(8,4))
        plt.plot(episodes_x, logs, label="Episode Reward", alpha=0.5)
        plt.plot(episodes_x, avg_reward_window_list,
                 label=f"Avg Reward (Last {metrics_window})", linewidth=2)
        plt.xlabel("Episode"); plt.ylabel("Reward")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"PathModel/reward_curves_ep{ep}{tag}.png", dpi=200)
        plt.close()

        # 到达率 + 匹配率
        plt.figure(figsize=(8,4))
        plt.plot(episodes_x, reach_rate_window_list,
                 label=f"Reach Rate (Last {metrics_window}) %", linewidth=2)
        plt.plot(episodes_x, match_rate_window_list,
                 label=f"Match Rate (Last {metrics_window}) %", linewidth=2)
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
            if in_random_exploration:
                a = np.random.randint(action_dim)
            else:
                a = agent.select_action(s_vec, evaluate=False)

            # Capture the pre-action neighbor for HER reward recomputation
            pre_action_neighbor = env.neighbor.copy() if hasattr(env, 'neighbor') else None

            ns, r, done, success = env.step(int(a))
            ns_vec = state_to_vector(ns)

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

        match_rate = env.match_ratio
        logs.append(ep_reward)
        success_list.append(success)
        match_list.append(match_rate)

        window = max(1, int(metrics_window))
        avg_reward_window = np.mean(logs[-window:])
        reach_rate_window = sum(success_list[-window:]) / len(success_list[-window:]) * 100
        match_rate_window = sum(match_list[-window:]) / len(match_list[-window:]) * 100

        avg_reward_window_list.append(avg_reward_window)
        reach_rate_window_list.append(reach_rate_window)
        match_rate_window_list.append(match_rate_window)

        actor_loss_ep = float(np.mean(ep_actor_losses)) if ep_actor_losses else np.nan
        critic_loss_ep = float(np.mean(ep_critic_losses)) if ep_critic_losses else np.nan

        actor_loss_ep_list.append(actor_loss_ep)
        critic_loss_ep_list.append(critic_loss_ep)

        if ep % 10 == 0:
            print(
                f"[Episode {ep:04d}] reward={ep_reward:.3f}, "
                f"average reward {window}={avg_reward_window:.3f}, "
                f"reach rate={reach_rate_window:.2f}%, "
                f"match rate={match_rate_window:.2f}% "
            )

        if ep % 1000 == 0:
            torch.save(agent.actor.state_dict(), f"PathModel/sac_actor_ep{ep}{tag}.pth")
            _save_metrics_and_plots(ep)

    episodes_x = np.arange(1, len(logs) + 1)


    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, logs, label="Episode Reward", alpha=0.5)
    plt.plot(episodes_x, avg_reward_window_list,
             label=f"Avg Reward (Last {metrics_window})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"PathModel/reward_curves{tag}.png", dpi=200)
    plt.close()

    # 图2: 到达率 + 匹配率（滑动窗口）
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, reach_rate_window_list,
             label=f"Reach Rate (Last {metrics_window}) %", linewidth=2)
    plt.plot(episodes_x, match_rate_window_list,
             label=f"Match Rate (Last {metrics_window}) %", linewidth=2)
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
        "avg_reward_window": avg_reward_window_list,
        "reach_rate_window": reach_rate_window_list,
        "match_rate_window": match_rate_window_list,
        "actor_loss": actor_loss_ep_list,
        "critic_loss": critic_loss_ep_list,
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

    # ========== 分层采样：平衡长距离样本比例 ==========
    # 长距离样本（20+ cells）原本只占约 2.4%，严重不足
    # 过采样长距离样本，使其在训练中出现更多次
    long_mask = shuffled_traj['distance_cells'] > 20
    long_traj = shuffled_traj[long_mask]
    short_traj = shuffled_traj[~long_mask]

    # 过采样倍数：长距离样本复制 10 倍，使占比从 ~2.4% 提升到 ~20%
    OVERSAMPLE_RATIO = 10
    if len(long_traj) > 0:
        long_traj_oversampled = pd.concat([long_traj] * OVERSAMPLE_RATIO, ignore_index=True)
        balanced_traj = pd.concat([short_traj, long_traj_oversampled], ignore_index=True)
    else:
        balanced_traj = shuffled_traj

    # 打乱顺序，确保长短距离混合
    balanced_traj = balanced_traj.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[Balanced Sampling] short={len(short_traj)}, long(oversampled {OVERSAMPLE_RATIO}x)={len(long_traj_oversampled) if len(long_traj) > 0 else 0}, total={len(balanced_traj)}")

    train_mode = True
    USE_GNN = False
    FOV = 5
    distance_threshold = 1.0
    env = PathEnv(train_mode=train_mode,
                  mapdata=mapdata,
                  traj=balanced_traj,
                  FOV=FOV,
                  distance_threshold=distance_threshold,
                  bfs_search_radius=30,
                  bfs_max_nodes=500000,
                  )

    # HER 配置：默认 0.8 (Future 策略)，可改为 0.0 关闭
    sac_cfg = SACConfig(her_prob=0.0, her_strategy='future')

    agent, logs = train_sac_on_pathenv(env, episodes=5000, cfg=sac_cfg,
                                       metrics_window=100, use_gnn=USE_GNN)
