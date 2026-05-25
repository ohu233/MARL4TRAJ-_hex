import os
import pickle
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from utils.Environment import ModeEnv
from utils.DQN import DQNConfig, DQNAgent, mode_state_to_vector

modelist = ['GSD', 'GG', 'TS', 'TG']
MODE_COLORS = {
    "TG": "orange",
    "GG": "blue",
    "GSD": "green",
    "TS": "red",
}

@dataclass
class TrainModeConfig:
    episodes: int = 8000
    max_episode_steps: int = 50
    seed: int = 42
    log_interval: int = 10
    save_interval: int = 1000
    save_dir: str = "ModeModel"
    save_name_prefix: str = "dqn_mode"
    metrics_window: int = 100


def train_dqn_on_modeenv(env: ModeEnv, cfg: DQNConfig, tcfg: TrainModeConfig):
    random.seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)

    os.makedirs(tcfg.save_dir, exist_ok=True)

    s0 = env.reset()
    env.traj_cnt -= 1
    s0_vec = mode_state_to_vector(s0)
    state_dim = s0_vec.shape[0]
    action_dim = 15  # 4bit mode组合 (1~15, 排除全零)

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, cfg=cfg)

    total_steps = 0
    reward_logs = []
    success_logs = []
    match_logs = []
    loss_logs = []
    mode_logs = []
    finish_logs = []
    true_mode_logs = []

    avg_reward_100_list = []
    succ_rate_100_list = []
    match_rate_100_list = []
    finish_rate_100_list = []
    mode_accuracy_100_list = []
    per_mode_finish_rate_100 = {m: [] for m in modelist}
    per_mode_accuracy_100 = {m: [] for m in modelist}

    log_path = os.path.join(tcfg.save_dir, "train_log.txt")
    log = open(log_path, "a", encoding="utf-8")

    for ep in range(1, tcfg.episodes + 1):
        s = env.reset()
        s_vec = mode_state_to_vector(s)

        ep_reward = 0.0
        ep_losses = []

        for _ in range(tcfg.max_episode_steps):
            total_steps += 1
            agent.total_steps = total_steps

            if total_steps < cfg.start_steps:
                a = np.random.randint(0, action_dim)
            else:
                a = agent.select_action(s_vec, evaluate=False)
                
            ns, r, done, succ, multi_match_rate = env.step(int(a))
            ns_vec = mode_state_to_vector(ns)

            # 真实标签 mode（若数据没有该列则给 None）
            if getattr(env, "current_row", None) is not None and "mode" in env.current_row.columns:
                true_mode = env.current_row["mode"].iat[0]
            else:
                true_mode = None

            agent.replay.push(s_vec, a, float(r), ns_vec, float(done))
            s_vec = ns_vec

            if total_steps % cfg.train_freq == 0:
                info = agent.update()
                if info:
                    ep_losses.append(info["loss"])

            agent.maybe_update_target()

            ep_reward += float(r)

            if done:
                break

        success_logs.append(succ)
        match_logs.append(multi_match_rate)

        mode_info = []
        # 当前所选modes
        for i in range(len(env.state['current']['mode'])):
            if env.state['current']['mode'][i] == 1:
                mode_info.append(modelist[i])
        # 真实mode在selected modes中的占比
        if true_mode in mode_info:
            mode_logs.append(1) # /len(mode_info))
        else:
            mode_logs.append(0)

        true_mode_logs.append(true_mode)
        finish_logs.append(env.finish)            
        reward_logs.append(ep_reward)
        loss_logs.append(float(np.mean(ep_losses)) if ep_losses else np.nan)

        w = tcfg.metrics_window
        avg_reward_100 = float(np.mean(reward_logs[-w:]))
        succ_rate_100 = float(np.mean(success_logs[-w:]) * 100.0)
        match_rate_100 = float(np.mean(match_logs[-w:]) * 100.0)
        mode_accuracy = float(np.mean(mode_logs[-w:]) * 100.0)
        mode_accuracy_100_list.append(mode_accuracy)
        avg_reward_100_list.append(avg_reward_100)
        succ_rate_100_list.append(succ_rate_100)
        match_rate_100_list.append(match_rate_100)
        finish_rate = float(np.mean(finish_logs[-w:]) * 100.0)
        finish_rate_100_list.append(finish_rate)

        # 按真实mode分组，计算窗口内每个mode的完成率和精确度
        true_mode_window = true_mode_logs[-w:]
        finish_window = finish_logs[-w:]
        mode_hit_window = mode_logs[-w:]
        for m in modelist:
            idx = [i for i, tm in enumerate(true_mode_window) if tm == m]
            if idx:
                per_mode_finish_rate_100[m].append(float(np.mean([finish_window[i] for i in idx]) * 100.0))
                per_mode_accuracy_100[m].append(float(np.mean([mode_hit_window[i] for i in idx]) * 100.0))
            else:
                per_mode_finish_rate_100[m].append(np.nan)
                per_mode_accuracy_100[m].append(np.nan)
        
        log.write(
            f"[Episode {ep:05d}] "
            f"reward = {ep_reward}, "
            f"average reward: {avg_reward_100:.3f}, "
            f"success_rate: {succ_rate_100:.3f}%, "
            f"match_rate: {match_rate_100:.3f}%\n"
            f"true_mode = {true_mode}, "
            f"selected mode = {mode_info}, "
            f"finish = {env.finish}, "
            f"step count = {env.step_cnt}, "
            f"mode accuracy = {mode_accuracy:.3f}%, "
            f"finish rate = {finish_rate:.3f}%\n"
            f"=============================================================================================================\n"
        )
        log.flush()

        if ep % tcfg.save_interval == 0:
            ckpt = os.path.join(tcfg.save_dir, f"{tcfg.save_name_prefix}_ep{ep}.pth")
            agent.save(ckpt)
            print(f"saved checkpoint: {ckpt}")

    # final save
    final_ckpt = os.path.join(tcfg.save_dir, f"{tcfg.save_name_prefix}_final.pth")
    agent.save(final_ckpt)
    print(f"saved final model: {final_ckpt}")

    # plots
    episodes_x = np.arange(1, len(reward_logs) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, reward_logs, label="Episode Reward", alpha=0.5)
    plt.plot(episodes_x, avg_reward_100_list, label="Avg Reward (Last 100)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Mode-DQN Training Reward Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(tcfg.save_dir, "reward_curves.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, succ_rate_100_list, label="Success Rate (Last 100) %", linewidth=2)
    plt.plot(episodes_x, match_rate_100_list, label="Match Rate (Last 100) %", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Rate (%)")
    plt.title("Mode-DQN Training Success/Match Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(tcfg.save_dir, "rate_curves.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, loss_logs, label="DQN Loss", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Mode-DQN Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(tcfg.save_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, finish_rate_100_list, label="ALL", color="black", linewidth=2.5)
    for mode in ["TG", "GG", "GSD", "TS"]:
        plt.plot(
            episodes_x,
            per_mode_finish_rate_100[mode],
            label=f"{mode}",
            color=MODE_COLORS[mode],
            linewidth=2,
        )
    plt.xlabel("Episode")
    plt.ylabel("Finish Rate (%)")
    plt.title("Mode-DQN Finish Rate Curves (ALL + Per Mode)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(tcfg.save_dir, "finish_rate_mode_curves.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, mode_accuracy_100_list, label="ALL", color="black", linewidth=2.5)
    for mode in ["TG", "GG", "GSD", "TS"]:
        plt.plot(
            episodes_x,
            per_mode_accuracy_100[mode],
            label=f"{mode}",
            color=MODE_COLORS[mode],
            linewidth=2,
        )
    plt.xlabel("Episode")
    plt.ylabel("Mode Accuracy (%)")
    plt.title("Mode-DQN Mode Accuracy Curves (ALL + Per Mode)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(tcfg.save_dir, "mode_accuracy_curves.png"), dpi=200)
    plt.close()

    metrics_df = pd.DataFrame({
        "episode": episodes_x,
        "episode_reward": reward_logs,
        "avg_reward_100": avg_reward_100_list,
        "success": success_logs,
        "succ_rate_100": succ_rate_100_list,
        "episode_match": match_logs,
        "match_rate_100": match_rate_100_list,
        "finish_rate_100": finish_rate_100_list,
        "mode_accuracy_100": mode_accuracy_100_list,
        "finish_rate_100_TG": per_mode_finish_rate_100["TG"],
        "finish_rate_100_GG": per_mode_finish_rate_100["GG"],
        "finish_rate_100_GSD": per_mode_finish_rate_100["GSD"],
        "finish_rate_100_TS": per_mode_finish_rate_100["TS"],
        "mode_accuracy_100_TG": per_mode_accuracy_100["TG"],
        "mode_accuracy_100_GG": per_mode_accuracy_100["GG"],
        "mode_accuracy_100_GSD": per_mode_accuracy_100["GSD"],
        "mode_accuracy_100_TS": per_mode_accuracy_100["TS"],
        "dqn_loss": loss_logs,
    })
    metrics_csv = os.path.join(tcfg.save_dir, "train_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8")
    print(f"saved metrics: {metrics_csv}")

    log.close()

    return agent, metrics_df


if __name__ == "__main__":
    with open("data/hex_grid.pkl", "rb") as f:
        mapdata = pickle.load(f)

    traj = pd.read_csv("data/data_lower_train_ordered.csv")

    # 兼容没有velocity列的数据
    if "velocity" not in traj.columns:
        traj["velocity"] = traj["distance"] / traj["time"].replace(0, np.nan)
        traj["velocity"] = traj["velocity"].fillna(0.0)

    env = ModeEnv(
        model_path="PathModel/sac_actor_ep5000_withConv_withCurri.pth",  # 已训练好的Path策略
        mapdata=mapdata,
        traj=traj,
        train_mode=True,
        fov=3,
        distance_threshold=1.0,
        use_conv=False,
    )

    dqn_cfg = DQNConfig(
        gamma=0.99,
        lr=1e-3,
        batch_size=128,
        buffer_size=200000,
        start_steps=2000,
        train_freq=1,
        target_update_interval=500,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=50000,
        hidden_dim=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    train_cfg = TrainModeConfig(
        episodes=5000,
        max_episode_steps=50,
        log_interval=10,
        save_interval=1000,
        save_dir="ModeModel",
        save_name_prefix="dqn_mode",
        metrics_window=100,
    )

    agent, metrics = train_dqn_on_modeenv(env, dqn_cfg, train_cfg)