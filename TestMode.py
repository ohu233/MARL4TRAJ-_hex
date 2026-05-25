import os
import pickle
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from utils.DQN import DQNAgent, DQNConfig, mode_state_to_vector
from utils.Environment import ModeEnv

MODE_LIST = ["GSD", "GG", "TS", "TG"]


@dataclass
class TestModeConfig:
    model_path: str = "ModeModel/dqn_mode_final.pth"
    traj_path: str = "data/data_lower_test.csv"
    map_path: str = "data/GridModesAdjacentRealworld.pkl"
    path_model_path: str = "PathModel/PathModel.pth"
    save_dir: str = "ModeModel/test_results"
    episodes: int = 0  # 0 表示使用测试集全量
    max_episode_steps: int = 50
    metrics_window: int = 100
    seed: int = 42


def _selected_modes_from_state(state):
    mode_vec = state.get("current", {}).get("mode", [0, 0, 0, 0])
    return [MODE_LIST[i] for i, v in enumerate(mode_vec) if int(v) == 1]


def evaluate_mode_dqn(cfg: TestModeConfig):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.save_dir, exist_ok=True)

    with open(cfg.map_path, "rb") as f:
        mapdata = pickle.load(f)

    traj = pd.read_csv(cfg.traj_path)
    if "velocity" not in traj.columns:
        traj["velocity"] = traj["distance"] / traj["time"].replace(0, np.nan)
        traj["velocity"] = traj["velocity"].fillna(0.0)

    env = ModeEnv(
        model_path=cfg.path_model_path,
        mapdata=mapdata,
        traj=traj,
        train_mode=False,
        fov=5,
        distance_threshold=1.0,
    )

    # 用一次 reset 推断 state_dim，然后把计数复位，保证从第 0 条样本开始评估
    s0 = env.reset()
    s0_vec = mode_state_to_vector(s0)
    env.traj_cnt = 0

    dqn_cfg = DQNConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    agent = DQNAgent(state_dim=s0_vec.shape[0], action_dim=15, cfg=dqn_cfg)
    agent.load(cfg.model_path)
    agent.q.eval()
    agent.q_tgt.eval()

    n_episodes = len(traj) if cfg.episodes <= 0 else min(cfg.episodes, len(traj))

    rewards = []
    successes = []
    finishes = []
    matches = []
    mode_hits = []
    step_counts = []

    rows = []

    log_path = os.path.join(cfg.save_dir, "test_log.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"Test episodes = {n_episodes}\n")
        log.write(f"Model = {cfg.model_path}\n")
        log.write("=" * 90 + "\n")

        for ep in range(1, n_episodes + 1):
            s = env.reset()
            s_vec = mode_state_to_vector(s)

            ep_reward = 0.0
            succ = 0
            multi_match_rate = 0.0

            for _ in range(cfg.max_episode_steps):
                a = agent.select_action(s_vec, evaluate=True)
                ns, r, done, succ, multi_match_rate = env.step(int(a))
                s_vec = mode_state_to_vector(ns)
                ep_reward += float(r)
                if done:
                    break

            true_mode = None
            if getattr(env, "current_row", None) is not None and "mode" in env.current_row.columns:
                true_mode = str(env.current_row["mode"].iat[0]).strip()

            selected_modes = _selected_modes_from_state(env.state)
            mode_hit = int(true_mode in selected_modes) if true_mode is not None else 0

            rewards.append(ep_reward)
            successes.append(int(succ))
            finishes.append(int(env.finish))
            matches.append(float(multi_match_rate))
            mode_hits.append(mode_hit)
            step_counts.append(int(env.step_cnt))

            w = cfg.metrics_window
            avg_reward_w = float(np.mean(rewards[-w:]))
            succ_rate_w = float(np.mean(successes[-w:]) * 100.0)
            finish_rate_w = float(np.mean(finishes[-w:]) * 100.0)
            mode_acc_w = float(np.mean(mode_hits[-w:]) * 100.0)
            match_rate_w = float(np.mean(matches[-w:]) * 100.0)

            log.write(
                f"[Episode {ep:05d}] "
                f"reward={ep_reward:.3f}, "
                f"avg_reward_{w}={avg_reward_w:.3f}, "
                f"succ_rate_{w}={succ_rate_w:.2f}%, "
                f"finish_rate_{w}={finish_rate_w:.2f}%, "
                f"mode_acc_{w}={mode_acc_w:.2f}%, "
                f"match_rate_{w}={match_rate_w:.2f}%\n"
                f"true_mode={true_mode}, selected_modes={selected_modes}, "
                f"success={int(succ)}, finish={env.finish}, steps={env.step_cnt}, "
                f"multi_match={multi_match_rate:.4f}\n"
                + "-" * 90
                + "\n"
            )

            print(
                f"[Ep {ep:05d}/{n_episodes}] "
                f"reward={ep_reward:.3f}, success={int(succ)}, finish={env.finish}, "
                f"steps={env.step_cnt}, true_mode={true_mode}, "
                f"selected={selected_modes}, mode_hit={mode_hit}"
            )

            rows.append(
                {
                    "episode": ep,
                    "reward": float(ep_reward),
                    "success": int(succ),
                    "finish": int(env.finish),
                    "step_count": int(env.step_cnt),
                    "true_mode": true_mode,
                    "selected_modes": "+".join(selected_modes),
                    "mode_hit": mode_hit,
                    "multi_match_rate": float(multi_match_rate),
                    "avg_reward_w": avg_reward_w,
                    "succ_rate_w": succ_rate_w,
                    "finish_rate_w": finish_rate_w,
                    "mode_acc_w": mode_acc_w,
                    "match_rate_w": match_rate_w,
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_csv = os.path.join(cfg.save_dir, "test_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8")

    episodes_x = np.arange(1, len(metrics_df) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, metrics_df["reward"], label="Episode Reward", alpha=0.4)
    plt.plot(episodes_x, metrics_df["avg_reward_w"], label=f"Avg Reward (Last {cfg.metrics_window})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Mode-DQN Test Reward Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "test_reward_curves.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, metrics_df["succ_rate_w"], label="Success Rate %", linewidth=2)
    plt.plot(episodes_x, metrics_df["finish_rate_w"], label="Finish Rate %", linewidth=2)
    plt.plot(episodes_x, metrics_df["mode_acc_w"], label="Mode Accuracy %", linewidth=2)
    plt.plot(episodes_x, metrics_df["match_rate_w"], label="Match Rate %", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Rate (%)")
    plt.title("Mode-DQN Test Metrics Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "test_metric_curves.png"), dpi=200)
    plt.close()

    print("=" * 60)
    print(f"Saved log     : {log_path}")
    print(f"Saved metrics : {metrics_csv}")
    print(f"Avg reward    : {np.mean(rewards):.3f}")
    print(f"Success rate  : {np.mean(successes) * 100.0:.2f}%")
    print(f"Finish rate   : {np.mean(finishes) * 100.0:.2f}%")
    print(f"Mode accuracy : {np.mean(mode_hits) * 100.0:.2f}%")
    print(f"Match rate    : {np.mean(matches) * 100.0:.2f}%")
    print("=" * 60)

    return metrics_df


if __name__ == "__main__":
    test_cfg = TestModeConfig(
        model_path="ModeModel/dqn_mode_final.pth",
        traj_path="data/data_lower_test.csv",
        map_path="data/GridModesAdjacentRealworld.pkl",
        path_model_path="PathModel/PathModel.pth",
        save_dir="ModeModel/test_results",
        episodes=0,
        max_episode_steps=50,
        metrics_window=100,
        seed=42,
    )

    evaluate_mode_dqn(test_cfg)