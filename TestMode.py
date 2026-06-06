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
    traj_path: str = "data/artificial_od_all.csv"
    map_path: str = "data/hex_grid.pkl"
    path_model_path: str = "PathModel/sac_actor_ep5000_withConv_withCurri.pth"
    save_dir: str = "ModeModel/test_results"
    episodes: int = 0  # 0 表示使用测试集全量
    metrics_window: int = 100
    seed: int = 42


def evaluate_mode_dqn(cfg: TestModeConfig):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.save_dir, exist_ok=True)

    with open(cfg.map_path, "rb") as f:
        mapdata = pickle.load(f)

    traj = pd.read_csv(cfg.traj_path)
    if "velocity" not in traj.columns:
        dist_col = "distance_m" if "distance_m" in traj.columns else "distance"
        traj["velocity"] = traj[dist_col] / traj["time"].replace(0, np.nan)
        traj["velocity"] = traj["velocity"].fillna(0.0)

    env = ModeEnv(
        model_path=cfg.path_model_path,
        mapdata=mapdata,
        traj=traj,
        train_mode=False,
        fov=5,
        distance_threshold=1.0,
        use_conv=False,
    )

    # 用一次 reset 推断 state_dim，然后把计数复位
    s0 = env.reset()
    s0_vec = mode_state_to_vector(s0)
    if hasattr(env, "reset_episode_order"):
        env.reset_episode_order()
    else:
        env.traj_cnt = 0

    dqn_cfg = DQNConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    agent = DQNAgent(state_dim=s0_vec.shape[0], action_dim=5, cfg=dqn_cfg)
    agent.load(cfg.model_path)
    agent.q.eval()
    agent.q_tgt.eval()

    n_episodes = len(traj) if cfg.episodes <= 0 else min(cfg.episodes, len(traj))

    rewards = []
    successes = []
    mode_correct_list = []
    true_mode_list = []
    survived_mode_list = []
    pred_mode_list = []
    stop_step_list = []

    per_mode_correct = {m: [] for m in MODE_LIST}

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

            stop_step = env.max_mode_steps
            for mode_step in range(1, env.max_mode_steps + 1):
                invalid_actions = set(np.where(env.invalid_action_mask())[0])
                a = agent.select_action(s_vec, invalid_actions=invalid_actions, evaluate=True)
                ns, r, done, succ = env.step(int(a))
                s_vec = mode_state_to_vector(ns)
                ep_reward += float(r)
                if int(a) == env.stop_action or done:
                    stop_step = mode_step
                if done:
                    break

            true_mode = None
            traj_id = ""
            if getattr(env, "current_row", None) is not None and "ID" in env.current_row.columns:
                traj_id = str(env.current_row["ID"].iat[0]).strip()
            if getattr(env, "current_row", None) is not None and "mode" in env.current_row.columns:
                true_mode = str(env.current_row["mode"].iat[0]).strip()

            survived_mode = env.active_modes[0] if len(env.active_modes) == 1 else ""
            pred_mode = env.state.get("pred_mode", "")
            mode_correct = int(pred_mode == true_mode) if true_mode in MODE_LIST else np.nan
            state = env.state
            match_rates = state.get("match_rates", [0, 0, 0, 0])
            multi_match = state.get("multi_match_rate", 0.0)
            delta_q = state.get("delta_q", [0, 0, 0, 0])
            belief = state.get("belief", [0, 0, 0, 0])

            rewards.append(ep_reward)
            successes.append(int(succ))
            mode_correct_list.append(mode_correct)
            true_mode_list.append(true_mode)
            survived_mode_list.append(survived_mode)
            pred_mode_list.append(pred_mode)
            stop_step_list.append(stop_step)

            if true_mode in per_mode_correct:
                per_mode_correct[true_mode].append(mode_correct)

            w = cfg.metrics_window
            avg_reward_w = float(np.mean(rewards[-w:]))
            succ_rate_w = float(np.mean(successes[-w:]) * 100.0)
            recent_correct = [v for v in mode_correct_list[-w:] if not np.isnan(v)]
            mode_acc_w = float(np.mean(recent_correct) * 100.0) if recent_correct else np.nan

            def fmt_arr(arr):
                return "[" + ", ".join(f"{float(v):.3f}" for v in arr) + "]"

            log.write(
                f"[Episode {ep:05d}] "
                f"reward={ep_reward:.3f}  "
                f"avg_reward_{{{w}}}={avg_reward_w:.3f}  "
                f"succ_rate_{{{w}}}={succ_rate_w:.2f}%  "
                f"mode_acc_{{{w}}}={mode_acc_w:.2f}%\n"
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
                + "-" * 90 + "\n"
            )

            print(
                f"[Ep {ep:05d}/{n_episodes}] "
                f"reward={ep_reward:.3f}, success={int(succ)}, "
                f"true_mode={true_mode}, pred={pred_mode}, "
                f"correct={mode_correct}"
            )

            rows.append({
                "episode": ep,
                "reward": float(ep_reward),
                "success": int(succ),
                "ID": traj_id,
                "true_mode": true_mode,
                "labeled": int(true_mode in MODE_LIST),
                "survived_mode": survived_mode,
                "pred_mode": pred_mode,
                "mode_correct": mode_correct,
                "stop_step": stop_step,
                "q_base": float(state.get("q_base", 0.0)),
                "multi_match": float(multi_match),
                "match_GSD": float(match_rates[0]),
                "match_GG": float(match_rates[1]),
                "match_TS": float(match_rates[2]),
                "match_TG": float(match_rates[3]),
                "belief_GSD": float(belief[0]),
                "belief_GG": float(belief[1]),
                "belief_TS": float(belief[2]),
                "belief_TG": float(belief[3]),
                "delta_q_GSD": float(delta_q[0]),
                "delta_q_GG": float(delta_q[1]),
                "delta_q_TS": float(delta_q[2]),
                "delta_q_TG": float(delta_q[3]),
                "active_modes": str(env.active_modes),
                "prev_modes": str(env.current_id_history),
            })

    metrics_df = pd.DataFrame(rows)
    metrics_csv = os.path.join(cfg.save_dir, "test_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8")

    episodes_x = np.arange(1, len(metrics_df) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, metrics_df["reward"], label="Episode Reward", alpha=0.4)
    plt.plot(episodes_x, metrics_df["reward"].rolling(cfg.metrics_window, min_periods=1).mean(),
             label=f"Avg Reward (Last {cfg.metrics_window})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Mode-DQN Test Reward Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "test_reward_curves.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_x, metrics_df["mode_correct"].rolling(cfg.metrics_window, min_periods=1).mean() * 100,
             label="Mode Accuracy %", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Rate (%)")
    plt.title("Mode-DQN Test Mode Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "test_accuracy_curves.png"), dpi=200)
    plt.close()

    print("=" * 60)
    print(f"Saved log     : {log_path}")
    print(f"Saved metrics : {metrics_csv}")
    print(f"Avg reward    : {np.mean(rewards):.3f}")
    print(f"Success rate  : {np.mean(successes) * 100.0:.2f}%")
    labeled_correct = [v for v in mode_correct_list if not np.isnan(v)]
    if labeled_correct:
        print(f"Mode accuracy : {np.mean(labeled_correct) * 100.0:.2f}%")
    else:
        print("Mode accuracy : N/A (unlabeled)")
    print("- Per mode accuracy:")
    for m in MODE_LIST:
        if per_mode_correct[m]:
            print(f"  {m}: {np.mean(per_mode_correct[m]) * 100.0:.2f}% ({len(per_mode_correct[m])} samples)")
    print("=" * 60)

    return metrics_df


if __name__ == "__main__":
    test_cfg = TestModeConfig(
        model_path="ModeModel/dqn_mode_final.pth",
        traj_path="data/artificial_od_all.csv",
        map_path="data/hex_grid.pkl",
        path_model_path="PathModel/sac_actor_ep5000_withConv_withCurri.pth",
        save_dir="ModeModel/test_results",
        episodes=0,
        metrics_window=100,
        seed=42,
    )

    evaluate_mode_dqn(test_cfg)
