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
    if hasattr(env, "reset_episode_order"):
        env.reset_episode_order()
    else:
        env.traj_cnt -= 1
    s0_vec = mode_state_to_vector(s0)
    state_dim = s0_vec.shape[0]
    action_dim = 5  # 0-3 remove mode, 4 stop

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, cfg=cfg)

    total_steps = 0
    reward_logs = []
    success_logs = []
    mode_correct_logs = []
    labeled_logs = []
    loss_logs = []
    traj_id_logs = []
    true_mode_logs = []
    survived_mode_logs = []
    pred_mode_logs = []
    stop_step_logs = []
    q_base_logs = []
    match_logs = {m: [] for m in modelist}
    belief_logs = {m: [] for m in modelist}
    delta_q_logs = {m: [] for m in modelist}

    avg_reward_100_list = []
    succ_rate_100_list = []
    mode_accuracy_100_list = []
    per_mode_accuracy_100 = {m: [] for m in modelist}

    log_path = os.path.join(tcfg.save_dir, "train_log.txt")
    log = open(log_path, "a", encoding="utf-8")

    for ep in range(1, tcfg.episodes + 1):
        s = env.reset()
        s_vec = mode_state_to_vector(s)

        ep_reward = 0.0
        ep_losses = []

        stop_step = env.max_mode_steps
        for mode_step in range(1, env.max_mode_steps + 1):
            total_steps += 1
            agent.total_steps = total_steps

            invalid_mask = env.invalid_action_mask()
            invalid_actions = set(np.where(invalid_mask)[0])

            if total_steps < cfg.start_steps:
                valid = [i for i in range(action_dim) if i not in invalid_actions]
                a = random.choice(valid) if valid else env.stop_action
            else:
                a = agent.select_action(s_vec, invalid_actions=invalid_actions, evaluate=False)

            ns, r, done, succ = env.step(int(a))
            ns_vec = mode_state_to_vector(ns)
            if int(a) == env.stop_action or done:
                stop_step = mode_step

            next_invalid_mask = env.invalid_action_mask()

            agent.replay.push(s_vec, a, float(r), ns_vec, float(done), next_invalid_mask)
            s_vec = ns_vec

            if total_steps % cfg.train_freq == 0:
                info = agent.update()
                if info:
                    ep_losses.append(info["loss"])

            agent.maybe_update_target()

            ep_reward += float(r)

            if done:
                break

        # Metrics
        traj_id = str(env.current_row['ID'].iat[0]).strip() if 'ID' in env.current_row.columns else ""
        true_mode = None
        if 'mode' in env.current_row.columns:
            candidate_mode = str(env.current_row['mode'].iat[0]).strip()
            if candidate_mode in modelist:
                true_mode = candidate_mode
        survived_mode = env.active_modes[0] if len(env.active_modes) == 1 else ""
        pred_mode = env.state.get("pred_mode", "")
        labeled = true_mode in modelist
        mode_correct = int(pred_mode == true_mode) if labeled else np.nan
        state = env.state
        match_rates = state.get("match_rates", [0, 0, 0, 0])
        multi_match = state.get("multi_match_rate", 0.0)
        delta_q = state.get("delta_q", [0, 0, 0, 0])
        belief = state.get("belief", [0, 0, 0, 0])

        true_mode_log = true_mode

        success_logs.append(succ)
        mode_correct_logs.append(mode_correct)
        labeled_logs.append(int(labeled))
        survived_mode_logs.append(survived_mode)
        pred_mode_logs.append(pred_mode)
        stop_step_logs.append(stop_step)
        q_base_logs.append(float(state.get("q_base", 0.0)))
        for idx, m in enumerate(modelist):
            match_logs[m].append(float(match_rates[idx]))
            belief_logs[m].append(float(belief[idx]))
            delta_q_logs[m].append(float(delta_q[idx]))
        traj_id_logs.append(traj_id)
        true_mode_logs.append(true_mode_log)
        reward_logs.append(ep_reward)
        loss_logs.append(float(np.mean(ep_losses)) if ep_losses else np.nan)

        w = tcfg.metrics_window
        avg_reward_100 = float(np.mean(reward_logs[-w:]))
        succ_rate_100 = float(np.mean(success_logs[-w:]) * 100.0)
        recent_correct = [v for v in mode_correct_logs[-w:] if not np.isnan(v)]
        mode_accuracy = float(np.mean(recent_correct) * 100.0) if recent_correct else np.nan
        mode_accuracy_100_list.append(mode_accuracy)
        avg_reward_100_list.append(avg_reward_100)
        succ_rate_100_list.append(succ_rate_100)

        # 按真实 mode 分组统计 accuracy
        true_mode_window = true_mode_logs[-w:]
        correct_window = mode_correct_logs[-w:]
        for m in modelist:
            idx_list = [i for i, tm in enumerate(true_mode_window) if tm == m]
            if idx_list:
                per_mode_accuracy_100[m].append(float(np.mean([correct_window[i] for i in idx_list]) * 100.0))
            else:
                per_mode_accuracy_100[m].append(np.nan)

        # 格式化数组
        def fmt_arr(arr):
            return "[" + ", ".join(f"{float(v):.3f}" for v in arr) + "]"

        log.write(
            f"[Episode {ep:05d}] "
            f"reward={ep_reward:.3f}  "
            f"avg_reward_{{{w}}}={avg_reward_100:.3f}  "
            f"succ_rate_{{{w}}}={succ_rate_100:.2f}%  "
            f"mode_acc_{{{w}}}={mode_accuracy:.2f}%\n"
            f"  ID={traj_id}  true={true_mode}  pred={pred_mode}  "
            f"correct={mode_correct}  q_base={state.get('q_base', 0.0):.3f}\n"
            f"  multi_match={float(multi_match):.3f}  "
            f"match_GSD={float(match_rates[0]):.3f}  "
            f"match_GG={float(match_rates[1]):.3f}  "
            f"match_TS={float(match_rates[2]):.3f}  "
            f"match_TG={float(match_rates[3]):.3f}\n"
            f"  delta_q = {fmt_arr(delta_q)}\n"
            f"  belief  = {fmt_arr(belief)}\n"
            f"  stop_step={stop_step}  active={env.active_modes}  "
            f"prev_modes={env.current_id_history}\n"
            f"{'='*100}\n"
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
    plt.plot(episodes_x, mode_accuracy_100_list, label="Mode Accuracy (Last 100) %", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Rate (%)")
    plt.title("Mode-DQN Training Curves")
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
        "mode_correct": mode_correct_logs,
        "mode_accuracy_100": mode_accuracy_100_list,
        "survived_mode": survived_mode_logs,
        "pred_mode": pred_mode_logs,
        "stop_step": stop_step_logs,
        "q_base": q_base_logs,
        "ID": traj_id_logs,
        "true_mode": true_mode_logs,
        "labeled": labeled_logs,
        "match_GSD": match_logs["GSD"],
        "match_GG": match_logs["GG"],
        "match_TS": match_logs["TS"],
        "match_TG": match_logs["TG"],
        "belief_GSD": belief_logs["GSD"],
        "belief_GG": belief_logs["GG"],
        "belief_TS": belief_logs["TS"],
        "belief_TG": belief_logs["TG"],
        "delta_q_GSD": delta_q_logs["GSD"],
        "delta_q_GG": delta_q_logs["GG"],
        "delta_q_TS": delta_q_logs["TS"],
        "delta_q_TG": delta_q_logs["TG"],
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

    traj = pd.read_csv("data/artificial_od_single.csv")

    # 兼容没有 velocity 列的数据
    if "velocity" not in traj.columns:
        dist_col = "distance_m" if "distance_m" in traj.columns else "distance"
        traj["velocity"] = traj[dist_col] / traj["time"].replace(0, np.nan)
        traj["velocity"] = traj["velocity"].fillna(0.0)

    env = ModeEnv(
        model_path="PathModel/sac_actor_ep5000.pth",
        mapdata=mapdata,
        traj=traj,
        train_mode=True,
        fov=5,
        distance_threshold=1.0,
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
        log_interval=10,
        save_interval=1000,
        save_dir="ModeModel",
        save_name_prefix="dqn_mode",
        metrics_window=100,
    )

    agent, metrics = train_dqn_on_modeenv(env, dqn_cfg, train_cfg)
